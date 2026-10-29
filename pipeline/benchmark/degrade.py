"""Make a DFL match's tracking look like ours: the profiles, the view the degrader sees and the random draws.

    view = tracking_view(match)                        # the tracking of a DflMatch, and nothing of its truth
    atd, info = degrade(view, "gaps", calibration, 0)  # an Atd of the BAR-ATH cache's layout, and what was drawn

What this is for. The benchmark's matches have the ball on every frame and one slot per player, and a pipeline
choice validated on that says little about ours: the ball is missing on 41% of our graded frames, in long gaps
(A01, A02), and the pipeline chooses what to do about exactly that. A profile is a recipe for making the DFL tracking
as bad as ours, and fidelity.py measures with the Calibration's own statistics functions how far it got. Profiles:

    clean     DFL's tracking as it is (inputs.clean_atd).
    gaps      the ball is missing where OUR ball was missing: our own missing-ball masks, transplanted (below).
              The ball's speed is NaN everywhere (our speed channel is NaN on 92% of the frames that have a ball: T31).
              The players are untouched.
    context   the tracking profile's players, exactly (the cap's draws included), and a ball whose gaps are placed where
              OURS were lost, by a hazard fitted on our own match, instead of transplanted from it: its pattern, not
              its clock (below).
    tracking  the gaps' ball, exactly, and the players broken up the way ours are: tracked spans and lost spans, and
              46 anonymous slots that tracklets take and leave; a wrong-team slot now and then, NaN speeds in runs and
              positions smoothed; and, since version 2, our frames with few players tracked, by a cap, no piece shorter
              than our shortest tracklet (MIN_PIECE_FRAMES) and the walk's tracked spans rescaled once (TRACKED_SCALE),
              all three below. Version 1, which R004 measured, had none of the three, and its three fidelity gates
              about how many slots are present were pending (they are gates now: fidelity.PENDING_GATES is empty).
    offpitch  the tracking profile's players, exactly, and its ball's transplanted gaps, thinned, with the ball hidden
              from when it is more than a margin past a line until it is back on the pitch, as ours is lost out of
              play (below).

The version of a profile (PROFILES) is part of what a report says about its tracking: change what a profile does and
bump it, so that two reports of different tracking never carry one version.

No truth. A DflMatch carries the answer (ball_status and ball_possession, which truth.py turns into H/A/D), and a
degradation that read either would quietly make the benchmark easier or harder in a way that ours is not. So the
degrader is handed a TrackingView, which has the tracking and the match's scalars and no flag of any kind; the checks
assert that no field of it is named status or possession, and that the inputs a degraded match is served with hold
nothing of the truth (test_degrade_never_sees_truth, test_degrade_serves_inputs_without_the_truth).

Randomness. rng_for(seed, match_index, component, half) is np.random.default_rng([seed, match_index,
COMPONENTS[component], half]): one stream per (seed, match, component, half), and no profile in the key. So:

* A draw never depends on the order the matches are run in, nor on which of them are (match_index is the registry's
  position, or a hash of the id for a match outside it: tracking_view); run one match and it gets what it got in the
  full run.
* The profiles nest. A later profile that adds the players' degradation has exactly this profile's ball, because the
  ball's draws are the "transplant" component's and the players' are other components'; the difference between two
  results is then one component at a time.
* Adding a component later does not move the draws of the earlier ones.

The transplant. Our missing-ball mask of each half (Calibration.ball_masks: True where the ball is missing, from the
kick-off to the end of the graded block, M frames) is copied onto the benchmark's half. A half of n frames takes the
mask of a source half, 1 or 2 with the same chance, from an offset drawn uniformly over 0..max(0, M - n): frame j of
the benchmark's half is missing where frame offset + j of the source half was. If the source runs out (a benchmark
half longer than M - offset) the mask goes on from the start of its last TAIL_FRAMES frames, round and round: what
follows is a stretch of the late half's pattern repeated, not a hole in the degradation. The mask is read from the
half's kick-off, as ours was measured (a benchmark half starts at its kick-off, and the Calibration measures ours from
there too), but its offset is drawn uniformly, so frame j of the benchmark's half has the gap pattern of the minute of
frame offset + j of ours: up to M - n frames later than its own, 7.6 minutes for a 68,000-frame half and our
79,467-frame first half. Coverage worsens through each half (A02), and a gap falls at another minute of the half than
the one it fell at in ours.

What the transplant keeps of ours, by construction: the gaps' durations, and the pattern of the mask over a stretch of n
frames of ours. It does not keep the clock (the stretch is not at the benchmark half's own minutes: the offset is drawn
uniformly, so it is up to M - n frames later, above) and it keeps the half's share missing only on average over the
offset, since coverage worsens through our halves (A02) and a stretch from early in the source half is better covered
than the half, one from late in it worse. What it does not keep: where the gaps are in the play. They are drawn
without looking at the DFL play, so
nothing that ties ours to it is in them: a dead ball is no likelier to be missing than any other frame (in the public
window ours is about three times as likely as a frame of the window in general, 33.6% of the dead frames against 11.8%
of all of them, A01), and where on the pitch a gap starts is the DFL ball's own place at a random frame, not ours (the
share of onsets within 2 m of a line differs from ours, 36.0% against 28.0% in R003; the direction is not the one the
plan guessed, and there is no base rate of ours to say what ours is relative to). That is the difference between this
profile and the `context` one (below), and it is why fidelity.py checks the share of dead frames with the ball
visible and the nearest player's agreement as warnings, and prints the statistic.

The ball is missing where the mask says so, and where DFL has no ball itself (a row with none stays NaN). Position x
and y are removed together, as an Atd requires (atd_cache._check_presence). Every array of a degraded Atd is read-only.

The context profile (version 1; context_probabilities, _context_ball, HAZARD_SCALE). The transplant copies the pattern
of our gaps, not their clock, and ignores the play; this profile places them where our tracker loses the ball: at the
lines, on slow balls and where players crowd round the ball (A07, A08). The players are the tracking profile's, exactly
(the same draws, the cap's included), and the ball's gaps are drawn from the hazard that the Calibration fitted on our
own match (hazard.py): the chance, on each frame, that the next frame starts a gap.

* The features (hazard.features) are those of the CLEAN DFL ball, visible on every row it has, and of the DEGRADED
  players (the slots of the Atd, NaN where nobody is), in metres from a pitch corner on the match's own pitch, for each
  half from its kick-off to the end of its graded block (the Calibration's spans), the minute counted from the kick-off.
  The ball is not masked yet when they are computed, so the speed on the first SPEED_STEPS frames after a gap is the
  true one, where in BAR-ATH's features it carries the speed from before the gap: an approximation, counted (the
  record's post_gap_rows over its at_risk_rows) and reported.
* The chance of a frame is hazard_c * HAZARD_SCALE * the model's probability, at most MAX_HAZARD_PROB (the capped rows
  are counted), and 0 on a frame with a NaN feature: no player present, no step yet to give a ball speed, or a ball that
  DFL does not have. Such rows were never at risk in the fit either (hazard_nan_rows). hazard_c is 1 on our own rows;
  the rate that carries to another match is HAZARD_SCALE, below.
* hazard.sample_mask draws the half's gaps from rng_for(seed, match_index, "hazard", half), the class of an onset being
  a line distance below NEAR_LINE_M at the at-risk frame (the frame before the gap, as calibration.ball_stats reads it),
  its third of the half the one of that frame's minute, and its length one of our own for that class and third,
  max(1, round(seconds * fps)) frames at the match's rate (context_durations). The mask is taken out of the ball by the
  gaps profile's own function (_remove_ball): x and y together, the speed NaN everywhere.
* What it keeps and what it loses, against `tracking`. It keeps where ours loses the ball (lines, slow balls, a crowd
  of players round the ball), and it ties the gaps to the DFL play, so a dead ball is as likely to be missing as the
  hazard makes it. It loses the coupling between the cap and the gaps: `tracking` reads the counts and the mask from one
  window of ours, so both follow our camera, and here the cap reads a window of ours and the gaps are drawn from the
  play. The hazard is a logistic model of six features fitted on 143 onsets of one match, and DFL tracks the ball out
  of play, so frames that ours has no ball for are at risk here at the rate that the model gives their features.

The offpitch profile (version 1; offpitch_masks, OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB). DFL's tracker keeps the ball
while it is out of play; ours loses it there. The ball is past a line on 17.68% of DFL's at-risk rows and on 4.07% of
ours (R005, R009), and P05's dead-ball model, trained on the tracking profile, splits first on where the ball is: past a
line, last or next seen past a line. It learned a cue that our match mostly does not show (T39), and the tracking
profile cannot say by how much that flatters it. This profile is the tracking profile exactly (the same players, the
cap's draws included, and the same transplant draws) with two changes to the ball, on each half's span from its
kick-off, in this order:

* Thinning (q, OFFPITCH_DROP_PROB; thin_mask). Each interior gap of the transplanted mask (a run of True that touches
  neither end of the span; an edge gap is cut by the span, as calibration.ball_stats counts it) is dropped whole, its
  frames given DFL's ball back, with probability q: one uniform per interior gap, in time order, from rng_for(seed,
  match_index, "thinning", half) (thinning_uniforms: a component of its own, the last, so that no earlier draw moves).
  An edge gap is never dropped. The transplanted mask already has our missing share, our losses out of play included,
  so the off-pitch mask on top of it counts those twice and the profile's missing share would be above ours: q takes
  it back down. The same uniforms serve every q, so a larger q drops a superset of a smaller one's gaps.
* Off the pitch (m, OFFPITCH_MARGIN_M), with hysteresis (offpitch_mask). The ball is hidden from the first frame on
  which DFL's clean ball is more than m metres past a touch or goal line (line_distance_m below -m: min(x, L - x, y,
  W - y) in metres on the match's own pitch, as hazard.features and calibration.ball_stats measure it) until the first
  later frame on which it is back on the pitch (a line distance of at least 0: a ball on a line is in play), which is
  visible again. Our tracker, once it has lost the ball off the pitch, finds it again when play comes back on. A
  frame-by-frame rule would make a ball held or rolling near -m flicker between hidden and visible, gaps and
  reappearances of a frame or two that ours does not have (gaps under 2 s hold 0.1% of our missing frames), and the
  hysteresis needs no parameter beyond m. A frame that DFL has no ball for neither starts nor ends a stretch, and a
  half that ends with the ball off the pitch stays hidden to its end. It draws nothing.

The mask is the union of the two, and the ball is taken out by the profile's shared _remove_ball (x and y together, the
speed NaN everywhere). The players are tracking's byte for byte, the cap's included: the cap reads the transplant's
choice (cap_limits), which thinning does not change. So a difference between a config's scores on tracking and on
offpitch is the ball's doing, and at q = 0 and an infinite margin the ball is tracking's too.

The two constants are fitted once, on label-free statistics, then frozen, as TRACKED_SCALE and HAZARD_SCALE were: m so
that the share of the visible ball frames that are past a line (line distance below 0), pooled over the 7 matches and 3
seeds, is ours (the same statistic on our kick-off spans), searched at q = 0; then q so that the pooled missing share is
the Calibration's, at that m; then the first share once more, and m refitted once at that q if it has moved by more than
0.005 (offpitch.py fit, R015). Until they are pinned (OFFPITCH_FIT holds the fit's record) the profile is refused.

What it assumes. m is fitted to a statistic of the frames where our ball IS seen, which is measurable. How long our ball
spends out of play in all is not (where our ball is during our gaps is unobservable), and the profile assumes that it is
like DFL's. R005 (2026-10-23) rejected a mask for the ball off the pitch as a mechanism whose parameter cannot be
measured on ours; Tian reopened that ruling on 2026-10-29 for this separate profile, on the ground that m is fitted to
the share of the visible frames past a line. The other remedy that R005 rejected, thinning the context hazard beyond a
line towards our near-line share, is not reopened: this profile has no hazard, and its near-line onset share is a check,
not a gate. tracking v3 stays the default.

The tracklets (tracking). In ours a slot is not a player: it is a tracklet, a stretch of one slot being present, which
ends when the tracker loses the player and is taken over by whoever it finds next (T15). DFL has one column per player.
The degrader breaks each player's time on the pitch into tracked spans and lost spans and gives each tracked span (a
tracklet) a slot, so that the slots appear, vanish and are reused. A slot's index is still not as empty of the player as
in ours: under the lowest-free rule (below) a player who comes back after a lost span lands in the slot he had 69.0% of
the time (860 of 1,247 returns, J03WMX, seed 0, with the wrong-team episodes on, as in version 1: R004), and a slot
idles between two tracklets for a median of 2.3 s (59.7% of its gaps last at most 3 s and 3.8% over 60 s), where ours
idles for 7.4 s (17.7% at most 3 s, 15.0% over 60 s: A09, R004). Before the episodes were added it was 76.0%, 2.0 s,
67.1% and none over 60 s (A09: version 1 drew those, with the episodes off; version 3's slot rule does not give them
again). No evidence source reads a slot's
identity today, beyond nearest_player's argmin on exact ties, which goes to the lower slot; but a method that follows a
slot across a loss would be validated optimistically here. The cap of version 2 (below) drops players and so breaks slot
identity further: test_data_tracking_profile_on_a_real_match_with_our_real_calibration prints the figures with it.

* What it draws from is ours, measured on our kick-off spans (Calibration): a tracked span's length is one of our
  tracklet lengths (tracklet_lengths_s, in frames at the match's rate, at least 1), drawn uniformly and multiplied by
  TRACKED_SCALE (below; rounded, at least 1 frame), so that their distribution is ours, TRACKED_SCALE times as long;
  and the lost spans are such that the share of
  a player's frames that are lost is d = 1 - the mean number of slots present over the frames with more than CAP = 18
  present, over 22 (present_counts, both halves). A lost span is geometric (integer, at least 1) with mean mu =
  mean_tracked * d / (1 - d), mean_tracked being the mean of the very lengths the walk draws: tracked and lost spans
  alternate, so the lost share tends to mu / (mean_tracked + mu) = d. The tail of our frames, those with 18 or fewer
  present, is left out of d, because it is another part of the profile's: counting it here would count it twice.
* The walk (walk_run) is per player and per contiguous run of the player's frames on the pitch (a frame has him when DFL
  has a position for him; a substitution or a dismissal ends a run). Each run starts with a tracked span, then a lost
  span, and so on; the last span is cut at the run's end. All the draws come from rng_for(seed, match_index,
  "tracklets", half), in the order: players by column, runs in time order, and within a run a tracked length and then
  a lost span. A d of 0 loses nobody, and then a run is one tracklet and nothing is drawn.
* The slots (choose_tracklets) are 46 as ours are: 24 for the home team (0 to 23) and 22 for the away team (24 to 45).
  The tracklets of a half are taken in order of (start frame, player column) and each takes the LOWEST FREE slot of its
  team's pool; a slot is free for a tracklet that starts at t only when its previous tracklet stopped before t: its
  half-open stop is at most t - 1, so that there is at least one empty row between two tracklets in a slot, and a slot
  never changes player between two adjacent rows (version 3; versions 1 and 2 let a tracklet take a slot on the very
  row it was freed: R004c measured 372 such handovers in version 2 over 21 runs, 17.7 a match and seed, 9 to 30 a
  run, each a jump of 13.8 m at the median, where ours has none). A pool with no free slot is a ValueError (a team
  has 11 players on the pitch and 22 or 24 slots, so it means that DFL gave positions to players who are not
  playing).
* The players cover each graded block, from its first row: not from the kick-off as the ball's mask does, because the
  mask's clock is a property of the gaps (coverage worsens through a half) and a player is tracked or lost whatever the
  clock. A DFL half starts at its kick-off, so the two are the same rows for the 7 matches. The rows between the halves
  have no players, as in DFL.
* The speed is a finite difference in m/s on the match's pitch: the step from the frame before, over the frame's
  duration, within a tracklet piece, and NaN on its first frame. DFL's own speed channel is not used. Then the NaN
  runs of ours are placed (below).
* The ball is exactly the gaps profile's: the same transplant draws. The profiles nest, so the difference between two
  of their results is one component.

The wrong-team episodes (T32). On 10.5% of our frames a team shows more than 11 slots while the other shows fewer: a
slot of one team carries a player of the other, because the tracker's identity followed the wrong player out of a duel.
Identity swaps happen where players come close, so the degrader cuts the walk's tracked spans with episodes of that
kind (episode_pieces), between the walk and the slot assignment: the walk's "tracklets" draws are not touched, and
the order of operations of a half is the tracklets, the episodes, the cap, the shortest-piece rule (both version 2,
below), the slot of every piece, the positions (smoothed), the speeds, the NaN runs.

* Onsets are a Bernoulli process over the rows of the half's graded block (the discrete-time Poisson process) at
  r = s * target / mean episode frames per row, target = Calibration.wrong_team_share, the episodes ours
  (Calibration.wrong_team_episodes_s, both teams in surplus together, as max(1, round(seconds * fps)) frames, their mean
  the mean of those lengths) and s = WRONG_TEAM_SCALE. All an onset needs is drawn up front from
  rng_for(seed, match_index, "wrong_team", half): the onset rows, one pick per onset, one duration per onset, each one
  call, so that the draws do not depend on which onsets are skipped.
* At an onset row the closest home-away pair of players tracked then is found in metres, by their true teams, the
  lowest (home column, away column) first on a tie. If it is within NEAR_PAIR_M (2 m) a pick chooses one of the two, and
  from the onset until the episode's end (or the end of his tracked span) his frames are a separate tracklet in the
  other team's pool; the rest of his span is a piece before and a new tracklet after (an identity switch). An onset
  with no pair that close is skipped (skipped_far), and so is one on a player who is already in an episode
  (skipped_overlap); both are counted, with the onsets, the episodes and the frames they cover.
* s is not 1: it was rescaled once, label-free, so that the share of frames that calibration.player_stats sees as
  having a wrong-team slot is ours (see WRONG_TEAM_SCALE). That share is lower than the share of frames in an episode.

The cap (version 2; cap_limits, cap_pieces). Our tracker has 18 or fewer slots present on 10.6% of its frames (5.2% have
14 or fewer): the camera is on a part of the pitch and the rest of the players are not tracked. The walk's loss d is
taken over the frames with more than CAP = 18 present and leaves these out, so the benchmark has none of them without
the cap. Between the episodes and the slots, on each frame of a half where OUR count k is at most CAP, only the k pieces
nearest the ball are kept and the rest are dropped on that frame; a piece that loses frames is cut into the pieces
between them, which are new tracklets and take the lowest free slot each (so a player who comes back after a drop may
not get his own slot back, as in ours, and the drops are counted). Where the counts come from: cap_limits reads
Calibration.present_counts[source half] through the SAME Transplant as the ball's mask of that half (same source half,
same offset, the same cyclic tail: transplant_counts and transplant read one window, _window), over the half's span from
its kick-off, so the cap and the ball's gaps come from the same frames of ours, which keeps what ties them (both follow
our camera); the counts are the frames of the masks, which a Calibration guarantees and cap_limits checks. The cap draws
nothing: no stream of its own, no draw of the other parts changes. Where the ball is: the clean DFL ball (view.ball_xy),
standing in for where our camera points, which follows the real ball even when our tracker loses it; on a frame where
DFL has none, its last place in the half, and before the first the centre spot (_ball_track). Membership is sticky (our
tracker makes no tracklet shorter than 8 frames, A09): within a run of frames at or below CAP the pieces kept stay kept
and membership changes only when the count requires it (it falls: the farthest go; it rises, or a kept piece ends: the
nearest present piece that is not kept comes in), never because two distances swap; at the run's end everybody is back.
With no frame at or below CAP the cap changes nothing: the players are the profile's without it, and the same holds with
the scale of the tracked spans at 1 and the shortest-piece rule off, set through the parameters, which are version 1's
parameters and make the walk, episodes, slots, speeds and smoothing of version 1 under version 3's slot rule, not
version 1's players (test_tracking_with_no_low_count_frame_is_the_players_without_the_cap).

The shortest piece and the scale of the tracked spans (version 2; MIN_PIECE_FRAMES, drop_short_pieces, TRACKED_SCALE).
Ours has no tracklet shorter than 8 frames (A09), and version 2 without this rule made 2.2% of its pieces so (906 of
40,866 pieces, R004b; 195 of them were already short before the cap, from the walk's run ends and the episodes), many of
them where our count steps across CAP and back, so after the cap and before the slots every piece shorter than
MIN_PIECE_FRAMES = 8 is dropped, whatever made it (the walk's run ends, an episode, the cap): the player is absent on
its frames, and the pieces, their slot-frames and the episode pieces among them are counted (short_pieces,
short_slot_frames, short_episode_pieces). The walk draws its tracked spans from our FINAL tracklets, which already hold
the cuts that the cap and the episodes add, so the profile counts our cuts twice, and its tracklet median would fall far
below ours (the walk alone 40.88 s against ours 41.28 s, with the episodes 29.26 s, with the cap 24.44 s, with both
19.68 s: R004b, with version 2's slot rule). The drawn lengths are therefore multiplied once by TRACKED_SCALE = 2.072
(lambda; the lost spans follow through mu, so the lost share stays d), worked out once from one label-free measurement
and frozen: the comment at the constant has the figures of versions 2 and 3 and the date. calibrated_parameters reads
the two constants when it is called (None is the constant) and a call can set them aside (tracked_scale=1.0 and
min_piece=0 are version 1's parameters; the players they make are the walk's with version 3's slot rule, not version
1's).

The NaN-speed runs (place_speed_nan): ours come in runs (7.8% of the present slot-frames, median 21 frames, p99 2,300).
After the speeds, runs are placed from the half's "speed_nan" stream: a present slot-frame chosen uniformly, a length
from our own runs, the next L frames of that slot NaN, stopping at its tracklet's end, until the half's NaN speeds
(the first rows of the tracklets, which are NaN already, included) reach Calibration.speed_nan_share. The ball's speed
stays NaN.

Smoothing. DFL's positions are rougher than ours (SMOOTH_WINDOW: the second difference is 27 times ours), so each
tracklet piece is smoothed with a centred moving average of that many frames, the window shrinking at the ends of the
piece to the rows inside it (smoothed); nothing crosses a piece. What is not matched, and cannot be on ours: a slowly
varying bias against the true position (README, limitations).

The record of the context profile's ball (info["ball"]) holds the scale and c and, per half and in all, counts: the rows
of the span, the rows where DFL has no ball, the NaN-feature rows (a ball and a feature that is undefined), the capped
rows, the gaps drawn (onsets) and the frames they take (missing_frames), the at-risk rows (a ball after the mask, every
feature defined, not the half's last row) and the at-risk rows within SPEED_STEPS frames after a gap. No frame.

The record of the offpitch profile's ball (info["ball"]) holds m (margin_m), q (drop_prob) and the Transplant of each
half, and, per half and in all, counts: the rows of the span, the transplanted mask's interior gaps, the gaps dropped
and the frames they gave back (frames_restored; the off-pitch mask may hide some of them again), the off-pitch stretches
and their frames (offpitch_frames; some may be in the transplanted mask already), and the frames of the final mask
(missing_frames; not counting the rows that DFL has no ball for outside it). No frame.

The record of what was drawn (info["players"]) holds counts and shares: the parameters (d, the mean tracked and lost
span, the scale, the targets, the smoothing window, the scale of the tracked spans as tracked_scale and the shortest
piece as min_piece_frames), per half the tracklets, the slots used, the frames on the pitch and the lost ones, the
episodes' counts, the NaN runs', the cap's (capped_frames, cap_slot_frames, cap_pieces_cut) and the shortest-piece
rule's (short_pieces, short_slot_frames, short_episode_pieces), and their totals. No frame.
"""

import hashlib
import math
import operator
from dataclasses import dataclass, field

import numpy as np

from pipeline.benchmark import calibration as calibration_module
from pipeline.benchmark import hazard
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import sources
from pipeline.inputs.atd_cache import Atd

# Params
CLEAN = "clean"
GAPS = "gaps"
TRACKING = "tracking"
CONTEXT = "context"
OFFPITCH = "offpitch"
# name -> version of how its tracking is made (reports say it). tracking 3: a slot is free from the row after its
# tracklet.
PROFILES = {CLEAN: 1, GAPS: 1, TRACKING: 3, CONTEXT: 1, OFFPITCH: 1}
# The version of the tracking profile that the context profile's players are built from: its players ARE tracking's, by
# the same code and the same draws, so a new version of tracking's players is a new version of context's. Pinned to
# PROFILES[TRACKING] by a check, recorded in context's info and in the reports of a context run (profile_record).
CONTEXT_TRACKING_VERSION = 3
# The same for the offpitch profile: its players, and the transplant draws its ball starts from, are tracking's.
OFFPITCH_TRACKING_VERSION = 3
# The streams of random numbers, by the part of the degradation that draws them. The number is the stream's place in the
# key of rng_for, so it never changes for a component and a new component takes the next one. The transplant, the
# tracklets, the wrong-team episodes and the NaN-speed runs are drawn so far; the hazard's is the context profile's and
# the thinning's the offpitch profile's.
COMPONENTS = {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4, "thinning": 5}
TAIL_FRAMES = 7500                  # a mask that runs out goes round the last 300 s of the source (25 Hz)
INDEX_BYTES = 4                     # an id outside the registry has the first 4 bytes of its sha256 as match_index
CAP = 18                            # d is taken over the frames with MORE than this many slots present (not the tail)
# The shortest tracklet of ours is 8 frames (0.32 s; A09, Calibration.tracklet_lengths_s): a piece shorter than this,
# whatever made it (the walk's run ends, an episode, the cap), is not a tracklet of ours, and the player is absent on
# its frames.
MIN_PIECE_FRAMES = 8
# lambda, the factor on the walk's drawn tracked lengths (calibrated_parameters). The walk draws its tracked spans from
# our FINAL tracklets, which already hold the cuts that the cap and the episodes add, so the profile counts our cuts
# twice: the walk alone gives a pooled tracklet median of 40.88 s against ours 41.28 s, and each part that cuts takes it
# further down (episodes 29.26 s, cap 24.44 s, both 19.68 s: R004b). It is rescaled ONCE, label-free, from one
# measurement: the pooled tracklet median of this profile at lambda = 1 with the shortest-piece rule in, seed 0, over
# the 7 DFL matches, laid end to end as fidelity.py pools it, against ours 41.28 s (Calibration). With version 2's slot
# rule (a tracklet could take a slot on the row it was freed, which merged two tracklets into one run of a slot) it was
# 20.22 s (13,192 tracklets), so lambda was 41.28 / 20.22 = 2.0415..., 2.042 to three decimals (2026-10-23, Task 14,
# R004b). The slot rule of version 3 (a slot is free from the row after its tracklet stopped: R004c) gives 19.92 s
# (13,396 tracklets), so lambda = 41.28 / 19.92 = 2.0723..., 2.072 (2026-10-23, Task 17, R004c): the bug that the slot
# rule fixed moved the measurement, so the constant was reopened once, and not for a gate, a WARN or a score. It is
# applied whether or not the tracklet-median gate would pass with it, and it is frozen: never touch it again because of
# a gate, a WARN or a score.
TRACKED_SCALE = 2.072
CENTRE_SPOT = 0.5                   # the cap's ball until DFL's has been seen in a half: the middle, in x and in y
FULL = calibration_module.FULL_COUNT    # 22: players on the pitch, so slots present on a frame where nobody is lost
HOME_SLOTS, AWAY_SLOTS = 24, 22     # our tracker's slots (the ATD metadata): the home team's first, then the away's
N_SLOTS = HOME_SLOTS + AWAY_SLOTS   # 46
DECIMALS = 9                        # the floats of a record are rounded to this, as calibration.py's are
TRACKING_FIELDS = ("present_counts", "tracklet_lengths_s", "wrong_team_episodes_s", "wrong_team_share",
                   "speed_nan_runs", "speed_nan_share")         # what the tracking profile reads of the Calibration
NEAR_PAIR_M = 2.0                   # an episode starts only where an opposing tracked pair is this close (inclusive)
CONTEXT_FIELDS = ("hazard_model", "hazard_c", "hazard_durations_s")  # what the context profile reads of the Calibration
MAX_HAZARD_PROB = 1.0 - 1e-9        # a chance of 1 is an infinite hazard, which hazard.sample_mask refuses: the cap
LINE_COLUMN = hazard.FEATURES.index("line_distance_m")    # the feature whose value below NEAR_LINE_M is the near class
CONTEXT_COUNTS = ("rows", "ball_missing_rows", "nan_feature_rows", "capped_rows", "onsets", "missing_frames",
                  "at_risk_rows", "post_gap_rows")      # the counts of info["ball"], per half and in all
# The factor on the hazard's chance (context_probabilities), on top of the Calibration's hazard_c, which is 1 on our own
# rows (it carries nothing to another match: the rate transfer is this). It calibrates a label-free statistic, the share
# of the frames without a ball, and was rescaled ONCE, on the odds: gaps start only from visible frames, so missing /
# visible is about linear in the hazard and the share is not, and a plain ratio would undershoot. At 1.0 the profile's
# missing share, on seed 0 over the 7 DFL matches pooled by sums, was 501,013 of 1,002,644 frames, 0.4997, against the
# Calibration's 0.4110 on its kick-off spans (+-0.02 is the gate), so the scale became 1.0 * [0.4110 / 0.5890] / [0.4997
# / 0.5003] = 0.6986, 0.699 to three decimals (2026-10-23, Task 17, R005). It is frozen: never touch it again because of
# a gate, a WARN or a score.
HAZARD_SCALE = 0.699
# The offpitch profile's two constants (offpitch_masks): m, the margin in metres past a line beyond which the ball is
# hidden until it is back on the pitch, and q, the chance that an interior gap of the transplanted mask is dropped. They
# calibrate two label-free statistics, the share of the visible ball frames past a line and the missing share, and are
# fitted ONCE by `offpitch.py fit` (R015), then pinned here with the fit's record (OFFPITCH_FIT: its scalars, which the
# two follow from) and frozen. None is not fitted yet, and the profile refuses to be made (_offpitch_constants).
OFFPITCH_MARGIN_M = None
OFFPITCH_DROP_PROB = None
OFFPITCH_FIT = None
# The counts of the offpitch profile's info["ball"], per half and in all.
OFFPITCH_COUNTS = ("interior_gaps", "gaps_dropped", "frames_restored", "offpitch_stretches", "offpitch_frames",
                   "missing_frames", "rows")
# s, the scale of the wrong-team onsets (episode_pieces). It calibrates a label-free statistic: the observable share of
# frames with a team in surplus (calibration.player_stats). That is less than the share of frames in an episode: a
# team shows more than 11 slots only while the other team's 11 are all tracked, and an episode of the other team at the
# same time puts it back at 11. At s = 1.0 the observable share, measured on seed 0 over the 7 matches pooled by sums
# (39,756 of 1,002,644 frames), was 0.03965, against our 0.10539 (+-0.03 is the gate), so s was rescaled ONCE, to
# 1.0 * 0.105391764 / 0.03965116 = 2.658 (2026-10-23, Task 13, R004). It is frozen: never touch it again because of a
# WARN, a gate or a score.
WRONG_TEAM_SCALE = 2.658
# The moving average's window over each tracklet piece, in frames (0 is none). The players' own jitter, the RMS of the
# second difference of a position, is 8.996 cm in the 7 DFL matches (clean players, pooled by sums of squares, from each
# kick-off to the end of the graded block) against 3.280 mm in ours, 27 times as much. The rule, set before measuring,
# was to smooth if DFL's is more than 2 x ours (Task 13, R004): it is, so every piece is smoothed with a 5-frame centred
# moving average. What the average leaves is reported by test_data_tracking_constants_follow_from_the_measurements.
SMOOTH_WINDOW = 5
MAX_SPEED_NAN_SHARE = 0.5           # the NaN-speed target is below this: a larger one may not be reachable
RUN_DRAWS_PER_FRAME = 20            # a NaN-speed loop gives up after this many run draws per present slot-frame


@dataclass(frozen=True, eq=False)
class TrackingView:
    """What the degrader sees of a match: the tracking and the scalars, and none of the truth.

    There is no ball_status and no ball_possession here, nor anything made of them. The arrays are the match's own
    (read-only when it is: dfl.load_match's are), shared and not copied, and are never changed: a degradation makes
    new ones. `match_index` is where the match's draws come from (rng_for); positions are normalised 0-1, speeds m/s,
    ball_z metres (DflMatch).
    """
    match_id: str
    match_index: int
    pitch: tuple
    fps: int
    graded: tuple                                    # ((half, start, stop), ...), half-open
    kickoffs: dict                                   # {half: first row}
    players_xy: np.ndarray = field(repr=False)       # (n, P, 2)
    players_speed: np.ndarray = field(repr=False)    # (n, P)
    player_team: np.ndarray = field(repr=False)      # (P,), 0 home, 1 away
    ball_xy: np.ndarray = field(repr=False)          # (n, 2)
    ball_z: np.ndarray = field(repr=False)           # (n,)
    ball_speed: np.ndarray = field(repr=False)       # (n,)

    @property
    def n_frames(self):
        return self.ball_xy.shape[0]


def index_of(match_id):
    """The match_index of an id: its place in the registry, or four bytes of its sha256 for an id outside it.

    A synthetic or a future match has no place in sources.MATCH_IDS, and must still always get the same draws, whatever
    other matches are run beside it: its index comes from the id alone.
    """
    if match_id in sources.MATCH_IDS:
        return sources.MATCH_IDS.index(match_id)
    return int.from_bytes(hashlib.sha256(match_id.encode("utf-8")).digest()[:INDEX_BYTES], "little")


def tracking_view(match, match_index=None):
    """The TrackingView of a DflMatch: its tracking and scalars; `match_index` defaults to index_of(match.match_id)."""
    index = index_of(match.match_id) if match_index is None else _whole("match_index", match_index)
    return TrackingView(
        match_id=match.match_id, match_index=index, pitch=tuple(match.pitch), fps=int(match.fps),
        graded=tuple(tuple(int(value) for value in block) for block in match.graded),
        kickoffs={int(half): int(row) for half, row in match.kickoffs.items()},
        players_xy=match.players_xy, players_speed=match.players_speed, player_team=match.player_team,
        ball_xy=match.ball_xy, ball_z=match.ball_z, ball_speed=match.ball_speed)


def rng_for(seed, match_index, component, half):
    """The generator of one component of one half of one match under one seed: independent of every other.

    np.random.default_rng([seed, match_index, COMPONENTS[component], half]); ValueError, naming the argument, for a
    component that is not in COMPONENTS and for a seed, an index or a half that is not a whole number of at least 0.
    """
    if component not in COMPONENTS:
        raise ValueError("component {!r} is not one of {}; a new part of the degradation takes the next number in "
                         "degrade.COMPONENTS".format(component, ", ".join(COMPONENTS)))
    return np.random.default_rng([_whole("seed", seed), _whole("match_index", match_index), COMPONENTS[component],
                                  _whole("half", half)])


def _whole(name, value):
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError("{} must be a whole number of at least 0, got {!r}".format(name, value)) from None
    if isinstance(value, bool) or number < 0:
        raise ValueError("{} must be a whole number of at least 0, got {!r}".format(name, value))
    return number


# -- the transplant ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Transplant:
    """Where a half's missing-ball mask comes from: the half of our match, and the frame of it that it starts at."""
    source_half: int
    offset: int


def choose_transplant(view, half, calibration, seed):
    """The Transplant of one half of a match: a source half 1 or 2, and an offset, drawn by rng_for(..., "transplant").

    The source half is uniform over the halves the Calibration has (1 and 2), and the offset uniform over 0..max(0, M -
    n), M the length of that half's mask and n the frames of the benchmark half from its kick-off to the end of its
    graded block. The draws are made in that order from one generator, so that the same (seed, match, half) always gets
    the same pair. ValueError for a half the view has no block for, and for a Calibration with no mask.
    """
    kickoff, stop = _span(view, half)
    n = stop - kickoff
    halves = sorted(calibration.ball_masks)
    if not halves:
        raise ValueError("the Calibration has no ball masks, so there is no gap of ours to transplant; build it with "
                         "calibration.load()")
    rng = rng_for(seed, view.match_index, "transplant", half)
    source_half = halves[int(rng.integers(len(halves)))]
    length = len(calibration.ball_masks[source_half])
    return Transplant(source_half=source_half, offset=int(rng.integers(max(0, length - n) + 1)))


def transplant(source, n, offset):
    """(n,) bool: `source` read from `offset`, round and round through its last TAIL_FRAMES frames once it ends.

    out[j] = source[offset + j] while offset + j is inside the source; from its end (index m), the mask goes on from
    the start of its tail: out[j] = source[m - t + (offset + j - m) % t] with t = min(TAIL_FRAMES, m). A fresh array.
    ValueError for a source that is not a non-empty 1-D bool array, and for an n or an offset below 0.
    """
    mask = np.asarray(source)
    if mask.dtype != np.bool_ or mask.ndim != 1 or mask.size == 0:
        raise ValueError("source must be a non-empty 1-D bool mask (True where the ball is missing), it is {} with "
                         "shape {}".format(mask.dtype, mask.shape))
    return mask[_window(mask.size, n, offset)]


def transplant_counts(source, n, offset):
    """(n,) read-only array: `source`, a list of whole numbers, read exactly as transplant reads a mask.

    The same frames (the same window, _window): from `offset`, and past the source's end round and round its last
    TAIL_FRAMES frames. It is how our slot counts, which the cap of the tracking profile reads, come from the frames
    that the ball's mask comes from. The result has the source's dtype and is a fresh array. ValueError for a source
    that is not a non-empty 1-D array of whole numbers, and for an n or an offset below 0.
    """
    counts = np.asarray(source)
    if counts.dtype.kind not in "iu" or counts.ndim != 1 or counts.size == 0:
        raise ValueError("source must be a non-empty 1-D array of whole numbers (the slots present on each frame), it "
                         "is {} with shape {}".format(counts.dtype, counts.shape))
    out = counts[_window(counts.size, n, offset)]
    out.setflags(write=False)
    return out


def _window(length, n, offset):
    """(n,) int64: the frames of a source of `length` frames that a window of n frames from `offset` reads.

    out[j] = offset + j while that is inside the source; from its end (index `length`), the window goes on from the
    start of the source's tail: length - t + (offset + j - length) % t with t = min(TAIL_FRAMES, length). ValueError for
    an n or an offset below 0 (not whole numbers)."""
    n, offset = _whole("n", n), _whole("offset", offset)
    tail = min(TAIL_FRAMES, length)
    index = offset + np.arange(n, dtype=np.int64)
    beyond = index >= length
    index[beyond] = length - tail + (index[beyond] - length) % tail
    return index


def _span(view, half):
    """(kick-off row, stop row) of a half of the view: the frames the ball's mask covers, as Calibration.ball_masks."""
    start, stop = _block(view, half)
    kickoff = view.kickoffs.get(half, start)
    if not start <= kickoff < stop:
        raise ValueError("{}: half {} kicks off at row {}, outside its block {}-{}".format(
            view.match_id, half, kickoff, start, stop))
    return kickoff, stop


def _block(view, half):
    """(first row, stop row) of the graded block of a half of the view: the frames the players' tracklets cover."""
    for number, start, stop in view.graded:
        if number == half:
            return start, stop
    raise ValueError("{}: no half {} in its graded blocks {}".format(
        view.match_id, half, ", ".join(str(block) for block in view.graded)))


# -- the tracklets ----------------------------------------------------------------------------------------------------

def _none():
    """What a field of frames is when the Calibration gives none: an empty read-only int64 array."""
    return _frames("nothing", [], allow_empty=True)


@dataclass(frozen=True, eq=False)
class TrackingParameters:
    """What the degradation draws from: our tracked spans, the lost share, and the episodes, NaN runs and smoothing.

    `lengths` are the tracked spans of ours (int64, each at least 1, read-only); a tracked span of the degraded match is
    one of them, drawn uniformly. `d` is the share of an on-pitch player's frames that are lost, `mean_tracked` the mean
    of `lengths` and `mu` = mean_tracked * d / (1 - d) the mean lost span (0 for d = 0: nobody is lost).

    The rest is each part's own, and neutral by default (none of the new parts: the walk's players alone): `episodes`
    are the wrong-team episodes of ours in frames (`mean_episode` their mean), `wrong_team_share` the share of frames
    that ours has a team in surplus on (the target), and `scale` is s: the onsets come at the rate
    onset_rate = s * wrong_team_share / mean_episode per row (0 with no scale, no share or no episodes). `nan_runs` are
    our runs of NaN speeds in frames and `speed_nan_share` the share of the present slot-frames to reach. `smooth` is
    the window in frames of the moving average over each tracklet piece (0 is none; otherwise odd, at least 3).
    `min_piece` is the shortest piece that is a tracklet, in frames (drop_short_pieces; 0 is no rule), and
    `tracked_scale` the factor that the lengths were multiplied by (calibrated_parameters), kept for the record: the
    lengths are already scaled.
    """
    lengths: np.ndarray = field(repr=False)
    d: float
    mean_tracked: float
    mu: float
    episodes: np.ndarray = field(default_factory=_none, repr=False)
    mean_episode: float = 0.0
    wrong_team_share: float = 0.0
    scale: float = 0.0
    nan_runs: np.ndarray = field(default_factory=_none, repr=False)
    speed_nan_share: float = 0.0
    smooth: int = 0
    min_piece: int = 0
    tracked_scale: float = 1.0

    @property
    def onset_rate(self):
        """The chance that a row of the half starts an episode: s * target / mean episode frames, 0 if there is none."""
        return self.scale * self.wrong_team_share / self.mean_episode if self.episodes.size else 0.0


@dataclass(frozen=True)
class Tracklet:
    """A stretch of one player being tracked as one slot: rows start .. stop - 1 of the match (half-open).

    The rows are not in its repr: under the cap a tracklet's cut points are where our slot count crossed CAP, in a
    window of ours, which is frame-level data of BAR-ATH and must not reach a log by an accident of printing a
    record."""
    player: int         # the column of the DFL match
    start: int = field(repr=False)
    stop: int = field(repr=False)
    slot: int           # 0 .. N_SLOTS - 1: the home team's slots first


def lost_share(present_counts):
    """d: 1 - (the mean number of slots present over the frames with more than CAP present) / 22, both halves pooled.

    `present_counts` is {half: counts} as Calibration.present_counts, the slots present on each frame of our kick-off
    spans. The frames with CAP or fewer present are our tracker's tail, which is not this part's, and are left out. A
    frame with all 22 present has nobody lost, so d is the share of an on-pitch player's frames that the tracker has
    lost, taken where it has most of them. The sum is of integers, so the figure is the same on any machine. ValueError
    for no counts, and for none above CAP.
    """
    arrays = [np.asarray(counts).ravel() for _, counts in sorted(present_counts.items())] if present_counts else []
    counts = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)
    kept = counts[counts > CAP]
    if kept.size == 0:
        raise ValueError("present_counts has no frame with more than {} slots present ({} frames in all), so d, the "
                         "share of an on-pitch player's frames that are lost, cannot be taken from it; pass the "
                         "Calibration of our own match, calibration.load()".format(CAP, counts.size))
    return 1.0 - int(kept.sum(dtype=np.int64)) / (kept.size * FULL)


def _frames(name, values, allow_empty=False):
    """A read-only int64 array of whole numbers of frames, each at least 1; ValueError naming `name` otherwise."""
    array = np.asarray(values)
    if array.ndim != 1 or (array.size == 0 and not allow_empty) or (array.size and array.dtype.kind not in "iu") or (
            array.size and (array < 1).any()):
        raise ValueError("{} must be a non-empty list of whole numbers of frames, each at least 1; got {!r}".format(
            name, array if array.size < 10 else array.shape))
    array = np.array(array, dtype=np.int64)
    array.setflags(write=False)
    return array


def _share(name, value, below, why):
    """`value` as a float in [0, below); ValueError naming `name` and saying `why` otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not (
            0 <= value < below):
        raise ValueError("{} must be at least 0 and below {:g}, {}; got {!r}".format(name, below, why, value))
    return float(value)


def tracking_parameters(lengths, d, episodes=None, wrong_team_share=0.0, scale=0.0, nan_runs=None,
                        speed_nan_share=0.0, smooth=0, min_piece=0, tracked_scale=1.0):
    """TrackingParameters of tracked spans `lengths` (whole frames, each at least 1) and a lost share d (0 to below 1).

    mu = mean_tracked * d / (1 - d). A lost span is at least 1 frame, so a mean below 1 cannot be drawn: a d so small
    that mu comes out below 1 (and above 0) is a ValueError that says so. d = 0 has mu = 0 and loses nobody.

    The other arguments are the parts that come after the tracklets, and are none by default. `episodes` (whole frames,
    each at least 1: wrong-team episodes of ours) and `wrong_team_share` (0 to below 1) with the `scale` s (at least 0)
    give the onsets' rate s * share / mean episode, which must be below 1; a share and a scale above 0 need episodes.
    `nan_runs` (whole frames, each at least 1) and `speed_nan_share` (0 to below MAX_SPEED_NAN_SHARE) are the NaN-speed
    runs and the share to reach, which needs runs. `smooth` is 0, or the odd window (at least 3) of the moving average.
    `min_piece` is 0 (no rule) or the shortest piece, in frames, that is a tracklet (drop_short_pieces), and
    `tracked_scale` a positive finite number, the factor already applied to `lengths` (the record's). ValueError, naming
    the argument, for anything else.
    """
    array = np.asarray(lengths)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu" or (array < 1).any():
        raise ValueError("lengths must be a non-empty list of whole numbers of frames, each at least 1, the tracked "
                         "spans of ours; got {!r}".format(array if array.size < 10 else array.shape))
    if isinstance(d, bool) or not isinstance(d, (int, float, np.floating)) or not 0 <= d < 1:
        raise ValueError("d must be at least 0 and below 1, the share of an on-pitch player's frames that are lost; "
                         "got {!r}".format(d))
    array = np.array(array, dtype=np.int64)
    array.setflags(write=False)
    mean = int(array.sum(dtype=np.int64)) / array.size
    mu = 0.0 if d == 0 else mean * d / (1.0 - d)
    if 0 < mu < 1:
        raise ValueError("a lost span is at least 1 frame, so its mean cannot be below 1, and d = {:.6g} with tracked "
                         "spans of mean {:.6g} frames asks for a mean of {:.6g}; the Calibration's d is of the order "
                         "of 0.02 and its tracked spans are thousands of frames".format(d, mean, mu))
    share = _share("wrong_team_share", wrong_team_share, 1, "the share of the frames that have a team in surplus")
    if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer, np.floating)) or not (
            math.isfinite(scale) and scale >= 0):
        raise ValueError("scale must be at least 0 and finite, the factor on the wrong-team onsets' rate; got {!r}"
                         .format(scale))
    shown = _frames("episodes", [] if episodes is None else episodes, allow_empty=episodes is None)
    if share > 0 and scale > 0 and not shown.size:
        raise ValueError("a wrong_team_share of {:g} at scale {:g} needs episodes, the wrong-team episodes of ours in "
                         "frames (the Calibration's wrong_team_episodes_s), and there are none".format(share, scale))
    mean_episode = int(shown.sum(dtype=np.int64)) / shown.size if shown.size else 0.0
    if shown.size and scale * share / mean_episode >= 1:
        raise ValueError("the onset rate {:g} = scale {:g} * wrong_team_share {:g} / mean episode {:g} frames must be "
                         "below 1: at most one episode starts on a row".format(
                             scale * share / mean_episode, scale, share, mean_episode))
    nan_share = _share("speed_nan_share", speed_nan_share, MAX_SPEED_NAN_SHARE,
                       "the share of the present slot-frames with a NaN speed")
    runs = _frames("nan_runs", [] if nan_runs is None else nan_runs, allow_empty=nan_runs is None)
    if nan_share > 0 and not runs.size:
        raise ValueError("a speed_nan_share of {:g} needs nan_runs, our NaN-speed runs in frames (the Calibration's "
                         "speed_nan_runs), and there are none".format(nan_share))
    if isinstance(smooth, bool) or not isinstance(smooth, (int, np.integer)) or smooth < 0 or (
            smooth and (smooth < 3 or smooth % 2 == 0)):
        raise ValueError("smooth must be 0 (none) or an odd number of frames, at least 3, the window of the moving "
                         "average; got {!r}".format(smooth))
    return TrackingParameters(lengths=array, d=float(d), mean_tracked=mean, mu=mu, episodes=shown,
                              mean_episode=mean_episode, wrong_team_share=share, scale=float(scale), nan_runs=runs,
                              speed_nan_share=nan_share, smooth=int(smooth), min_piece=_min_piece(min_piece),
                              tracked_scale=_tracked_scale(tracked_scale))


def _min_piece(value):
    """`value` as a whole number of frames, at least 0; ValueError naming min_piece otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("min_piece must be a whole number of frames, at least 0 (0 is no rule): the shortest piece "
                         "that is a tracklet; got {!r}".format(value))
    return int(value)


def _tracked_scale(value):
    """`value` as a float, positive and finite; ValueError naming tracked_scale otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not (
            math.isfinite(value) and value > 0):
        raise ValueError("tracked_scale must be a positive finite number, the factor on the walk's drawn tracked "
                         "lengths; got {!r}".format(value))
    return float(value)


def calibrated_parameters(calibration, fps, profile=TRACKING, tracked_scale=None, min_piece=None):
    """TrackingParameters from the Calibration: its tracklet lengths and episodes at the match's rate, d from its slots.

    A tracked span is max(1, round(seconds * fps)) frames, `fps` the match's own (DFL's 25), multiplied by
    `tracked_scale` (lambda: the walk draws its spans from our final tracklets, which already hold the cuts that the cap
    and the episodes add, so a rescale corrects it once, R004b) and rounded again, at least 1 frame; the mean lost span
    follows through mu (the lost share stays d). `min_piece` is the shortest piece that is a tracklet. None for either
    is the module's constant, TRACKED_SCALE and MIN_PIECE_FRAMES, read here when the function is called (as
    WRONG_TEAM_SCALE and SMOOTH_WINDOW are), and not bound when the module loads. d is lost_share over present_counts.
    A wrong-team episode is converted the same way (wrong_team_episodes_s, both directions together), and the mean
    episode is the mean of those very lengths; the NaN-speed runs are already in frames (speed_nan_runs). The targets
    are the Calibration's wrong_team_share and speed_nan_share, the scale is WRONG_TEAM_SCALE and the smoothing window
    SMOOTH_WINDOW. ValueError, naming the profile, for a Calibration that lacks any of TRACKING_FIELDS (the made-up
    ones of the checks, a Calibration of an older layout), for no tracklets, for a share above 0 with no episodes or
    runs to draw, and for an fps that is not a whole number, at least 1.
    """
    tracked_scale = TRACKED_SCALE if tracked_scale is None else tracked_scale
    min_piece = MIN_PIECE_FRAMES if min_piece is None else min_piece
    absent = [name for name in TRACKING_FIELDS if not hasattr(calibration, name)]
    if absent:
        raise ValueError("profile {!r} needs the Calibration's {}, and this one has no {}; pass calibration.load()"
                         .format(profile, ", ".join(TRACKING_FIELDS), " and no ".join(absent)))
    if _whole("fps", fps) < 1:
        raise ValueError("fps must be a whole number of at least 1, got {!r}".format(fps))
    seconds = np.asarray(calibration.tracklet_lengths_s, dtype=np.float64)
    if seconds.ndim != 1 or seconds.size == 0 or not np.isfinite(seconds).all():
        raise ValueError("the Calibration's tracklet_lengths_s must be a non-empty list of finite lengths in seconds, "
                         "its tracked spans; it has shape {}".format(seconds.shape))
    lengths = np.maximum(1, np.rint(seconds * fps)).astype(np.int64)
    lengths = np.maximum(1, np.rint(lengths * _tracked_scale(tracked_scale))).astype(np.int64)
    episodes_s = np.asarray(calibration.wrong_team_episodes_s, dtype=np.float64)
    if episodes_s.ndim != 1 or not np.isfinite(episodes_s).all():
        raise ValueError("the Calibration's wrong_team_episodes_s must be a list of finite lengths in seconds, its "
                         "wrong-team episodes; it has shape {}".format(episodes_s.shape))
    if not episodes_s.size and calibration.wrong_team_share > 0:
        raise ValueError("the Calibration's wrong_team_share is {:g} but its wrong_team_episodes_s is empty: there are "
                         "no episodes of ours to draw the lengths from".format(calibration.wrong_team_share))
    episodes = np.maximum(1, np.rint(episodes_s * fps)).astype(np.int64) if episodes_s.size else None
    runs = np.asarray(calibration.speed_nan_runs)
    if not runs.size and calibration.speed_nan_share > 0:
        raise ValueError("the Calibration's speed_nan_share is {:g} but its speed_nan_runs is empty: there are no runs "
                         "of ours to draw the lengths from".format(calibration.speed_nan_share))
    return tracking_parameters(lengths, lost_share(calibration.present_counts), episodes=episodes,
                               wrong_team_share=calibration.wrong_team_share, scale=WRONG_TEAM_SCALE,
                               nan_runs=runs if runs.size else None, speed_nan_share=calibration.speed_nan_share,
                               smooth=SMOOTH_WINDOW, min_piece=min_piece, tracked_scale=tracked_scale)


def walk_run(rng, length, parameters):
    """[(start, stop)] tracked spans of a run of `length` frames on the pitch, as offsets in it (half-open, plain ints).

    The run starts with a tracked span: one of parameters.lengths, drawn uniformly (rng.integers). A lost span follows,
    geometric with mean mu (rng.geometric(1 / mu): integer, at least 1), then a tracked span again, and so on; the span
    that reaches past the run's end is cut there, and a lost span that does is the end of the walk. A run that is no
    longer than its first tracked span is one tracklet. With mu = 0 (d = 0) nobody is lost: the run is one tracklet,
    and nothing is drawn.
    """
    if parameters.mu == 0:
        return [(0, int(length))] if length > 0 else []
    lengths, count, p = parameters.lengths, parameters.lengths.size, 1.0 / parameters.mu
    spans, row = [], 0
    while row < length:
        stop = min(row + int(lengths[int(rng.integers(count))]), length)
        spans.append((row, stop))
        if stop >= length:
            break
        row = stop + int(rng.geometric(p))
    return spans


def walk_spans(view, half, parameters, seed):
    """[(first, player, last)] tracked spans of one half, sorted: the walk (walk_run) on every run of every player.

    The walk is made on each player's runs of rows with a position, over the half's graded block, with
    rng_for(seed, view.match_index, "tracklets", half): players by column, runs in time order. A span is the rows
    first .. last - 1 of the match (half-open). ValueError, naming the half, for one the view has no block for.
    """
    start, stop = _block(view, half)
    present = ~np.isnan(np.asarray(view.players_xy)[start:stop, :, 0])
    rng = rng_for(seed, view.match_index, "tracklets", half)
    spans = []
    for player in range(present.shape[1]):
        firsts, stops = _runs(present[:, player])
        for run_start, run_stop in zip(firsts.tolist(), stops.tolist()):
            for first, last in walk_run(rng, run_stop - run_start, parameters):
                spans.append((start + run_start + first, player, start + run_start + last))
    spans.sort()
    return spans


def episode_pieces(view, half, spans, parameters, seed):
    """([(first, player, last, pool)], counts): a half's spans cut by its wrong-team episodes; pool is a team, 0 or 1.

    A piece is carried by the pool's slots: the player's own team's, unless the piece is an episode, which is carried
    by the other team's. With no onset rate (a scale or a share of 0, or no episodes) the pieces are the spans, each in
    its player's own pool, and nothing is drawn. Otherwise, from rng_for(seed, view.match_index, "wrong_team", half),
    in this order and each as one call: one uniform per row of the half's graded block (a row with a uniform below the
    rate is an onset), then one pick per onset (0 is the home player of the pair below, 1 the away one) and one duration
    per onset (an index into the episodes). So what an onset draws never depends on which onsets are skipped.

    The onsets are taken in time order. At an onset row t the pair is the closest home-away pair, in metres on the
    match's pitch, among the players tracked at t (inside a span), by their true teams (view.player_team), the lowest
    (home column, away column) first on a tie. If it is no farther than NEAR_PAIR_M (inclusive) the pick chooses one of
    its two players, who is then carried by the other team's pool for rows t .. min(t + duration, end of the piece
    holding t) - 1, and the piece splits in up to three: before t (own pool), the episode, and after it (own pool, a
    new tracklet: the split is an identity switch). If the piece holding t is already an episode, the onset is skipped
    (skipped_overlap); if there is no pair within NEAR_PAIR_M, or one team has nobody tracked at t, it is skipped
    (skipped_far). The counts are onsets, episodes, skipped_far, skipped_overlap and episode_frames (the rows carried
    by the other pool).
    """
    team = np.asarray(view.player_team)
    counts = {"onsets": 0, "episodes": 0, "skipped_far": 0, "skipped_overlap": 0, "episode_frames": 0}
    own = {}
    for first, player, last in spans:
        own.setdefault(player, []).append([first, last, False])
    rate = parameters.onset_rate
    if rate > 0:
        start, stop = _block(view, half)
        rng = rng_for(seed, view.match_index, "wrong_team", half)
        onsets = start + np.flatnonzero(rng.random(stop - start) < rate)
        picks = rng.integers(2, size=onsets.size)
        durations = parameters.episodes[rng.integers(parameters.episodes.size, size=onsets.size)]
        counts["onsets"] = int(onsets.size)
        source = np.asarray(view.players_xy, dtype=np.float64)
        pitch = np.asarray(view.pitch, dtype=np.float64)
        held = np.zeros((onsets.size, source.shape[1]), dtype=bool)         # who is tracked at each onset
        for first, player, last in spans:
            held[np.searchsorted(onsets, first):np.searchsorted(onsets, last), player] = True
        for k, t in enumerate(onsets.tolist()):
            tracked = np.flatnonzero(held[k])
            home, away = tracked[team[tracked] == 0], tracked[team[tracked] == 1]
            if not home.size or not away.size:
                counts["skipped_far"] += 1
                continue
            where = source[t] * pitch
            gap = np.hypot(where[home][:, None, 0] - where[away][None, :, 0],
                           where[home][:, None, 1] - where[away][None, :, 1])
            row, column = divmod(int(np.argmin(gap)), away.size)         # the first minimum: the lowest columns
            if gap[row, column] > NEAR_PAIR_M:
                counts["skipped_far"] += 1
                continue
            player = int((home[row], away[column])[int(picks[k])])
            index = next(i for i, (first, last, _) in enumerate(own[player]) if first <= t < last)
            first, last, inside = own[player][index]
            if inside:
                counts["skipped_overlap"] += 1
                continue
            end = min(t + int(durations[k]), last)
            own[player][index:index + 1] = ([[first, t, False]] if t > first else []) + [[t, end, True]] + (
                [[end, last, False]] if end < last else [])
            counts["episodes"] += 1
            counts["episode_frames"] += end - t
    pieces = sorted((first, player, last, int(1 - team[player]) if inside else int(team[player]))
                    for player, parts in own.items() for first, last, inside in parts)
    return pieces, counts


def cap_limits(view, calibration, seed):
    """{half: (n,) int16 read-only}: how many slots of ours were present on each frame of the window that a half reads.

    Each half takes the Transplant of its ball's mask (choose_transplant: the same source half and offset, for it is the
    same call of a fresh generator of the same key) and reads Calibration.present_counts[source half] through it
    (transplant_counts), over the half's span from its kick-off, n = stop - kick-off frames. So the cap that these
    counts give and the gaps of the ball come from the same frames of ours, which keeps what ties them: both follow our
    camera. ValueError for a Calibration whose counts of a half are missing or are not as long as its ball mask (a
    Calibration's counts are the frames of its masks: calibration.load checks it), naming the half and both lengths.
    """
    limits = {}
    for half, _, _ in view.graded:
        kickoff, stop = _span(view, half)
        choice = choose_transplant(view, half, calibration, seed)
        source = choice.source_half
        if source not in calibration.present_counts:
            raise ValueError("half {}: the Calibration's present_counts has no half {} (it has {}), which the "
                             "ball's mask of this half was transplanted from; pass calibration.load()".format(
                                 half, source, ", ".join(str(number) for number in sorted(calibration.present_counts))
                                 or "none"))
        counts, mask = np.asarray(calibration.present_counts[source]), np.asarray(calibration.ball_masks[source])
        if counts.shape != mask.shape:
            raise ValueError("half {}: the Calibration's present_counts of half {} has {} frames and its "
                             "ball_masks {}: a Calibration's counts are the frames of its masks, and the cap and the "
                             "ball read one window of both; pass calibration.load()".format(
                                 half, source, counts.size, mask.size))
        limits[half] = transplant_counts(counts, stop - kickoff, choice.offset)
    return limits


def _ball_track(view, half):
    """(n, 2) float64: where the cap takes the ball to be on each row of a half's graded block, normalised.

    The clean DFL ball (view.ball_xy), standing in for where our camera points, which follows the real ball even when
    our tracker loses it; on a row where DFL has none, its last place in the half; before the first, the centre spot."""
    start, stop = _block(view, half)
    ball = np.asarray(view.ball_xy, dtype=np.float64)[start:stop]
    seen = np.isfinite(ball).all(axis=1)
    last = np.maximum.accumulate(np.where(seen, np.arange(stop - start), -1))
    return np.where((last >= 0)[:, None], ball[np.maximum(last, 0)], CENTRE_SPOT)


def cap_pieces(view, half, pieces, limit):
    """([(first, player, last, pool)], counts): a half's pieces without the players that our counts say were not
    tracked.

    `limit` is the half's cap_limits (n counts, from its kick-off to the end of its graded block), or None for no cap.
    On a row where the count k is at most CAP, only k of the pieces present are kept and the rest are dropped on that
    row; a piece that loses rows is cut into the pieces between them (each a new tracklet, which assign_slots gives the
    lowest free slot: a drop breaks a slot's identity, as ours does). Rows before the kick-off, and rows with a count
    above CAP, are not touched; nothing is drawn.

    Membership is sticky. A run is a maximal stretch of rows with a count of at most CAP. At the run's first row the k
    pieces nearest the ball are kept; at every later row the kept pieces that are still present stay, and if they are
    more than k the farthest of them are dropped, if fewer the nearest present pieces that are not kept come in; so
    membership changes only when the count requires it, never because two distances swap. At the run's end every piece
    present is kept again, and the next run starts from nothing. Nearest is by the distance in metres to the ball
    (_ball_track), ties to the lower player column; the piece's position on a row is its player's.

    The counts are {"capped_frames": rows with a count of at most CAP, "cap_slot_frames": slot-frames dropped,
    "cap_pieces_cut": pieces that lost a row}. ValueError, naming the half, for a limit that is not a count of each row
    of the half or not made of whole numbers.
    """
    counts = {"capped_frames": 0, "cap_slot_frames": 0, "cap_pieces_cut": 0}
    if limit is None:
        return pieces, counts
    kickoff, stop = _span(view, half)
    limit = np.asarray(limit)
    if limit.shape != (stop - kickoff,):
        raise ValueError("half {}: the cap's counts must be one for each of the {} rows from the kick-off to the end "
                         "of the block, and there are {}".format(half, stop - kickoff, limit.size))
    if limit.dtype.kind not in "iu":
        raise ValueError("half {}: the cap's counts must be whole numbers (the slots present on each row), they are {}"
                         .format(half, limit.dtype))
    capped = limit <= CAP
    counts["capped_frames"] = int(capped.sum())
    if not capped.any() or not pieces:
        return pieces, counts
    start, _ = _block(view, half)
    firsts = np.array([piece[0] for piece in pieces], dtype=np.int64)
    lasts = np.array([piece[2] for piece in pieces], dtype=np.int64)
    players = np.array([piece[1] for piece in pieces], dtype=np.int64)
    source = np.asarray(view.players_xy, dtype=np.float64)
    pitch = np.asarray(view.pitch, dtype=np.float64)
    ball = _ball_track(view, half)
    dropped = {}
    for run_first, run_stop in zip(*(found.tolist() for found in _runs(capped))):
        a, b = kickoff + run_first, kickoff + run_stop
        near = np.flatnonzero((firsts < b) & (lasts > a))
        if not near.size:
            continue
        rows = np.arange(a, b)
        here = (firsts[near][None, :] <= rows[:, None]) & (rows[:, None] < lasts[near][None, :])
        away = (source[a:b][:, players[near]] - ball[a - start:b - start][:, None, :]) * pitch
        distance = np.where(here, np.hypot(away[:, :, 0], away[:, :, 1]), np.inf)
        kept = np.zeros(near.size, dtype=bool)
        for step in range(b - a):
            k = int(limit[a - kickoff + step])
            present = here[step]
            kept &= present
            order = np.lexsort((players[near], distance[step]))     # nearest first, ties to the lower player column
            order = order[present[order]]
            have = kept[order]
            if have.sum() > k:
                kept[order[have][k:]] = False
            elif have.sum() < k:
                kept[order[~have][:k - int(have.sum())]] = True
            for index in near[present & ~kept]:
                dropped.setdefault(int(index), []).append(a + step)
    out = []
    for index, (first, player, last, pool) in enumerate(pieces):
        gone = dropped.get(index)
        if not gone:
            out.append((first, player, last, pool))
            continue
        counts["cap_pieces_cut"] += 1
        counts["cap_slot_frames"] += len(gone)
        alive = np.ones(last - first, dtype=bool)
        alive[np.array(gone, dtype=np.int64) - first] = False
        out += [(first + low, player, first + high, pool) for low, high in zip(*(f.tolist() for f in _runs(alive)))]
    return sorted(out), counts


def drop_short_pieces(view, pieces, minimum):
    """([(first, player, last, pool)], counts): the pieces of at least `minimum` frames, and what was dropped.

    Ours has no tracklet shorter than MIN_PIECE_FRAMES (A09), so a piece shorter than `minimum` is dropped whatever made
    it (the walk's run ends, an episode, the cap): the player is absent on its frames, and it takes no slot. The rest
    come back as they went in, in their order; 0 is no rule. The counts are {"short_pieces": pieces dropped,
    "short_slot_frames": the frames they held, "short_episode_pieces": how many of them were in the other team's pool
    (an episode)}. ValueError for a `minimum` that is not a whole number of at least 0.
    """
    minimum = _min_piece(minimum)
    counts = {"short_pieces": 0, "short_slot_frames": 0, "short_episode_pieces": 0}
    if minimum == 0:
        return pieces, counts
    team = np.asarray(view.player_team)
    kept = []
    for piece in pieces:
        first, player, last, pool = piece
        if last - first >= minimum:
            kept.append(piece)
            continue
        counts["short_pieces"] += 1
        counts["short_slot_frames"] += last - first
        counts["short_episode_pieces"] += int(pool != int(team[player]))
    return kept, counts


def assign_slots(view, half, pieces):
    """[Tracklet] of the pieces (first, player, last, pool), in order of (start, player), each in the lowest free slot.

    Each piece takes the lowest slot of its pool's team (home 0-23, away 24-45) that its previous tracklet has left, at
    least one row before: a slot is free for a piece that starts at t only when its last tracklet stopped before t (its
    half-open stop is at most t - 1), so that two tracklets of a slot are never on adjacent rows and a slot never
    changes player between two frames. A tracklet that took the slot on the row it was freed would put the position of
    another player into the slot's trajectory (versions 1 and 2 did: R004c counted 372 over 21 runs in version 2,
    17.7 a match and seed, with a median jump of 13.8 m). ValueError, naming the match, the half and the row, for a
    pool with no free slot.
    """
    free_from = [-1] * N_SLOTS
    tracklets = []
    for first, player, last, pool in pieces:
        home = pool == 0
        slots = range(HOME_SLOTS) if home else range(HOME_SLOTS, N_SLOTS)
        slot = next((candidate for candidate in slots if free_from[candidate] < first), None)
        if slot is None:
            raise ValueError("{}: half {}: no free {} slot for the player in column {}, whose tracklet starts at frame "
                             "{}: the {} slots of its pool are all taken by players tracked at that frame or the one "
                             "before. A team has 11 players on the pitch, so look at where the match's players_xy "
                             "gives a position to a player who is not playing".format(
                                 view.match_id, half, "home" if home else "away", player, first, len(slots)))
        free_from[slot] = last
        tracklets.append(Tracklet(player=player, start=first, stop=last, slot=slot))
    return tracklets


@dataclass(frozen=True, eq=False)
class HalfPlayers:
    """One half's players as slots: the tracklets in order of (start, player) and the episodes' counts (counts only).

    The tracklets are not in its repr (the same reason as Tracklet's rows: the cap's cut points are frame-level
    data)."""
    tracklets: list = field(repr=False)
    counts: dict


def choose_half(view, half, parameters, seed, limit=None):
    """HalfPlayers of one half: the walk's spans (walk_spans), cut by the wrong-team episodes (episode_pieces), capped
    by our counts (cap_pieces, with `limit`, the half's cap_limits: none is no cap), without the pieces shorter than
    parameters.min_piece (drop_short_pieces), slotted.

    The order is the one the degradation documents: the per-player tracklets, the episodes that split them, the cap that
    drops players, the pieces that are too short to be tracklets, then the slot of every piece (assign_slots), so that a
    player who comes back after a drop takes the lowest free slot. ValueError as walk_spans, cap_pieces and
    assign_slots.
    """
    pieces, counts = episode_pieces(view, half, walk_spans(view, half, parameters, seed), parameters, seed)
    pieces, capped = cap_pieces(view, half, pieces, limit)
    pieces, short = drop_short_pieces(view, pieces, parameters.min_piece)
    return HalfPlayers(tracklets=assign_slots(view, half, pieces), counts={**counts, **capped, **short})


def choose_tracklets(view, half, parameters, seed):
    """[Tracklet] of one half of a match, in order of (start, player), each with the lowest free slot of its pool.

    The tracklets of choose_half, with no cap: with no wrong-team episodes (the parameters' defaults) they are the
    walk's spans, each in the lowest free slot of its pool (version 3's rule: a slot is free from the row after its last
    tracklet stopped).
    """
    return choose_half(view, half, parameters, seed).tracklets


def smoothed(run, window):
    """`run` (m, 2) as its centred moving average over `window` rows: the window shrinks at the ends to the rows inside.

    A fresh array of the same length. Every row is the mean of the rows within window // 2 of it that the run has, so a
    run of one row is itself and nothing outside the run is read: the pieces of a tracklet are smoothed one by one.
    ValueError for a window that is not an odd whole number of at least 1 (1 is the identity).
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1 or window % 2 == 0:
        raise ValueError("window must be an odd whole number of frames, at least 1; got {!r}".format(window))
    run = np.asarray(run, dtype=np.float64)
    reach, rows = window // 2, run.shape[0]
    if reach == 0 or rows < 2:
        return np.array(run)
    total = np.concatenate([np.zeros((1, run.shape[1])), np.cumsum(run, axis=0)])
    index = np.arange(rows)
    low, high = np.maximum(index - reach, 0), np.minimum(index + reach + 1, rows)
    return (total[high] - total[low]) / (high - low)[:, None]


def place_speed_nan(view, half, xy, speed, tracklets, parameters, seed):
    """Put the NaN-speed runs into `speed` (in place) for one half; {"speed_nan_runs", "speed_nan_frames"}.

    Present slot-frames are the half's graded block's rows where `xy` has a position, listed in time order and by slot
    within a row. A run starts at one of them chosen uniformly and takes its length from parameters.nan_runs (one
    draw each, in this order, from rng_for(seed, view.match_index, "speed_nan", half)), marks the next L rows of that
    slot NaN and stops at the end of its tracklet. Runs are placed one after another until the NaN speeds of the
    half's present slot-frames, the first rows of the tracklets included (they are NaN already), are at least
    parameters.speed_nan_share of them: the last run overshoots by at most its own length. A target of 0 places none.
    ValueError for a target that the draws do not reach in RUN_DRAWS_PER_FRAME per present slot-frame.
    """
    start, stop = _block(view, half)
    present = np.isfinite(xy[start:stop, :, 0])
    marked = present & np.isnan(speed[start:stop])
    total = int(present.sum())
    runs = 0
    if parameters.speed_nan_share > 0 and total:
        rows, slots = np.nonzero(present)
        rng = rng_for(seed, view.match_index, "speed_nan", half)
        pieces = {}
        for t in tracklets:
            pieces.setdefault(t.slot, []).append((t.start, t.stop))
        edges = {slot: np.array(sorted(found)).T for slot, found in pieces.items()}
        count, goal = int(marked.sum()), parameters.speed_nan_share * total
        while count < goal:
            if runs >= RUN_DRAWS_PER_FRAME * total:
                raise ValueError("{}: half {}: {} NaN-speed runs have not reached a share of {:g} of the {} present "
                                 "slot-frames; the target is too large for the runs to cover".format(
                                     view.match_id, half, runs, parameters.speed_nan_share, total))
            index = int(rng.integers(total))
            length = int(parameters.nan_runs[int(rng.integers(parameters.nan_runs.size))])
            row, slot = start + int(rows[index]), int(slots[index])
            firsts, lasts = edges[slot]
            end = min(row + length, int(lasts[np.searchsorted(firsts, row, side="right") - 1]))
            seen = marked[row - start:end - start, slot]
            count += int((~seen).sum())
            seen[:] = True
            speed[row:end, slot] = np.nan
            runs += 1
    return {"speed_nan_runs": runs, "speed_nan_frames": int(marked.sum())}


def degraded_players(view, parameters, seed, limits=None):
    """(players_xy, players_speed, tracklets, counts): the match's players as slots, the tracklets {half: [Tracklet]}
    and the counts {half: {...}} of what was drawn for each half (_players_record).

    `limits` is {half: cap_limits of the half}, the counts of ours that cap the players present; None is no cap. The
    players are then those of the same parameters without the cap: with version 1's parameters (tracked_scale 1,
    min_piece 0) the walk, episodes, slots, speeds and smoothing of version 1 under version 3's slot rule, and with the
    profile's own the profile's players without the cap.

    players_xy is (n, N_SLOTS, 2) and players_speed (n, N_SLOTS), NaN where no tracklet has the slot; a tracklet's slot
    holds the player's position on its rows, smoothed by parameters.smooth if it is not 0 (smoothed: inside the piece),
    and its speed there is the step from the frame before in m/s on the match's pitch, NaN on the tracklet's first row.
    Then the NaN-speed runs (place_speed_nan). Rows outside the graded blocks have no players. Read-only.
    """
    source = np.asarray(view.players_xy, dtype=np.float64)
    pitch = np.asarray(view.pitch, dtype=np.float64)
    xy = np.full((view.n_frames, N_SLOTS, 2), np.nan)
    speed = np.full((view.n_frames, N_SLOTS), np.nan)
    records, counts = {}, {}
    for half, _, _ in view.graded:
        chosen = choose_half(view, half, parameters, seed, None if limits is None else _limit_of(limits, half))
        records[half] = chosen.tracklets
        for t in chosen.tracklets:
            run = source[t.start:t.stop, t.player]
            if parameters.smooth:
                run = smoothed(run, parameters.smooth)
            xy[t.start:t.stop, t.slot] = run
            if t.stop - t.start > 1:
                step = np.diff(run * pitch, axis=0)
                speed[t.start + 1:t.stop, t.slot] = np.hypot(step[:, 0], step[:, 1]) * view.fps
        counts[half] = {**chosen.counts, **place_speed_nan(view, half, xy, speed, chosen.tracklets, parameters, seed)}
    for array in (xy, speed):
        array.setflags(write=False)
    return xy, speed, records, counts


def _limit_of(limits, half):
    """The cap's counts of a half, or a ValueError that names the half and the halves there are."""
    if half not in limits:
        raise ValueError("no cap's counts for half {}; limits has {}".format(
            half, ", ".join(str(number) for number in sorted(limits)) or "none"))
    return limits[half]


def slot_team():
    """(N_SLOTS,) int8, read-only: 0 for the home team's slots (the first 24), 1 for the away team's (the last 22)."""
    team = np.array([0] * HOME_SLOTS + [1] * AWAY_SLOTS, dtype=np.int8)
    team.setflags(write=False)
    return team


def _runs(mask):
    """(first, stop) of the maximal runs of True in a 1-D bool array: int64 arrays, half-open, in time order."""
    padded = np.zeros(mask.size + 2, dtype=np.int8)
    padded[1:-1] = mask
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _players_record(view, parameters, records, counts):
    """The JSON-ready account of the players of every half: the parameters and, per half, counts. No frame in it."""
    halves, on_pitch, lost, count = {}, 0, 0, 0
    sums = dict.fromkeys(("onsets", "episodes", "skipped_far", "skipped_overlap", "episode_frames", "speed_nan_runs",
                          "speed_nan_frames", "capped_frames", "cap_slot_frames", "cap_pieces_cut", "short_pieces",
                          "short_slot_frames", "short_episode_pieces"), 0)
    for half, start, stop in view.graded:
        frames = int((~np.isnan(np.asarray(view.players_xy)[start:stop, :, 0])).sum())
        tracked = sum(t.stop - t.start for t in records[half])
        halves[half] = {"tracklets": len(records[half]), "slots_used": len({t.slot for t in records[half]}),
                        "on_pitch_frames": frames, "lost_frames": frames - tracked,
                        **{name: int(counts[half][name]) for name in sums}}
        on_pitch, lost, count = on_pitch + frames, lost + frames - tracked, count + len(records[half])
        for name in sums:
            sums[name] += halves[half][name]
    return {"d": round(parameters.d, DECIMALS), "mean_tracked_frames": round(parameters.mean_tracked, DECIMALS),
            "mean_lost_frames": round(parameters.mu, DECIMALS), "tracklets": count, "on_pitch_frames": on_pitch,
            "lost_frames": lost, "lost_share": round(lost / on_pitch, DECIMALS) if on_pitch else None,
            "scale": parameters.scale, "wrong_team_target": round(parameters.wrong_team_share, DECIMALS),
            "mean_episode_frames": round(parameters.mean_episode, DECIMALS),
            "onset_rate": round(parameters.onset_rate, DECIMALS + 3), "smooth_window": parameters.smooth,
            "tracked_scale": parameters.tracked_scale, "min_piece_frames": parameters.min_piece,
            "speed_nan_target": round(parameters.speed_nan_share, DECIMALS), **sums,
            "speed_nan_share": (round(sums["speed_nan_frames"] / (on_pitch - lost), DECIMALS)
                                if on_pitch > lost else None),
            "halves": {half: halves[half] for half in sorted(halves)}}


# -- the profiles -----------------------------------------------------------------------------------------------------

def degrade(view, profile, calibration, seed):
    """(Atd, info): the tracking of `view` as `profile` makes it under `seed`, and what was drawn to make it.

    `calibration` is the Calibration of BAR-ATH (calibration.load()): the gaps, tracking and context profiles need it,
    the clean one does not use it. The Atd has the BAR-ATH cache's own layout (inputs.clean_atd), every array read-only,
    and the length of the match; the tracking and context profiles' have 46 slots (slot_team) where the others have one
    per DFL player.
    `info` is JSON-ready and holds no frame of anybody: {"profile", "version", "seed", "match_id", "match_index"} and,
    for gaps and tracking, "transplants" ({half: {"source_half", "offset", "frames"}}) and "mask_share": the share of
    the frames that the transplant covers where the transplanted mask says the ball is missing. It is the mask's
    share, not the ball's after it: a ball that DFL itself has no frame for is missing as well. For tracking also
    "players": the parameters and the counts of the tracklets, episodes and NaN runs (_players_record). For context,
    the same "players" (the tracking profile's, byte for byte), "tracking_version" (the version of tracking's players
    that they are built from: CONTEXT_TRACKING_VERSION), "transplants" (the draws that its cap reads: no mask is
    transplanted) and "ball": the scale, c and the counts of the hazard's draws (_context_ball). For offpitch, the same
    "players", "tracking_version" (OFFPITCH_TRACKING_VERSION), "transplants" (tracking's, which its ball's mask starts
    from), "ball" (m, q, the Transplants and the counts of offpitch_masks) and "mask_share" (the share of the spans'
    rows that its masks take). ValueError for a profile that is not in PROFILES, for a seed that is not a whole number
    of at least 0, for a profile that needs the Calibration without one (or, for tracking, without the fields it reads:
    calibrated_parameters; for context, without CONTEXT_FIELDS, or with a hazard that does not go: its hazard_go is
    False, A08), for offpitch while its constants are not fitted (_offpitch_constants), and for a pool of slots with no
    free slot (choose_tracklets).
    """
    if profile not in PROFILES:
        raise ValueError("profile {!r} is not one of: {}".format(profile, ", ".join(sorted(PROFILES))))
    seed = _whole("seed", seed)
    atd = match_inputs.clean_atd(view)
    info = {"profile": profile, "version": PROFILES[profile], "seed": seed, "match_id": view.match_id,
            "match_index": view.match_index}
    if profile == CLEAN:
        return atd, info
    if calibration is None:
        raise ValueError("profile {!r} takes its ball gaps{} from our own match, so it needs the Calibration; pass "
                         "calibration.load()".format(profile, " and its tracklets" if profile in (
                             TRACKING, CONTEXT, OFFPITCH) else ""))
    if profile == CONTEXT:
        absent = [name for name in CONTEXT_FIELDS if not hasattr(calibration, name)]
        if absent:
            raise ValueError("profile {!r} needs the Calibration's {}, and this one has no {}; pass calibration.load()"
                             .format(profile, ", ".join(CONTEXT_FIELDS), " and no ".join(absent)))
        # A Calibration always has the field (a test double of one may not, and is not judged on it).
        if getattr(calibration, "hazard_go", True) is False:
            raise ValueError("profile {!r} places the ball's gaps by the hazard fitted on our match, and this "
                             "Calibration's hazard_go is False: a cross-half AUC of the full model is below {} (A08), "
                             "so the model does not rank the frames before a gap above the others and would place "
                             "them by nothing that ours learned. Use the {} profile (the default), or measure again "
                             "with calibration.load(rebuild=True) and read A08 before trusting this one".format(
                                 profile, calibration_module.HAZARD_GO_AUC, TRACKING))
    elif profile == OFFPITCH:
        masks, info["ball"] = offpitch_masks(view, calibration, seed, *_offpitch_constants())
        ball_xy, ball_speed = _remove_ball(view, atd, masks)
        info["transplants"] = {half: dict(record) for half, record in info["ball"]["transplants"].items()}
        rows = info["ball"]["rows"]
        info["mask_share"] = info["ball"]["missing_frames"] / rows if rows else None
    else:
        ball_xy, ball_speed, info["transplants"], info["mask_share"] = _gap_ball(view, atd, calibration, seed)
    if profile == GAPS:
        players = (atd.players_xy, atd.players_speed, atd.slot_team)
    else:
        parameters = calibrated_parameters(calibration, view.fps, profile)
        xy, speed, records, counts = degraded_players(view, parameters, seed, cap_limits(view, calibration, seed))
        info["players"] = _players_record(view, parameters, records, counts)
        players = (xy, speed, slot_team())
        if profile == CONTEXT:
            info["tracking_version"] = CONTEXT_TRACKING_VERSION
            info["transplants"] = _transplant_records(view, calibration, seed)
            ball_xy, ball_speed, info["ball"], info["mask_share"] = _context_ball(view, atd, xy, calibration, seed)
        elif profile == OFFPITCH:
            info["tracking_version"] = OFFPITCH_TRACKING_VERSION
    degraded = Atd(players_xy=players[0], players_speed=players[1], ball_xy=ball_xy, ball_speed=ball_speed,
                   slot_team=players[2])
    return degraded, info


def profile_record(profile):
    """{"name", "version"} of a profile as the reports of a run state it; context's and offpitch's also have
    "tracking_version".

    That is the version of the tracking profile that their players are built from (CONTEXT_TRACKING_VERSION,
    OFFPITCH_TRACKING_VERSION): a report of a context or offpitch run says it beside the profile's own, so that it names
    the players' version as well as the ball's. KeyError for a name that is not in PROFILES (the callers have
    checked)."""
    record = {"name": profile, "version": PROFILES[profile]}
    if profile == CONTEXT:
        record["tracking_version"] = CONTEXT_TRACKING_VERSION
    elif profile == OFFPITCH:
        record["tracking_version"] = OFFPITCH_TRACKING_VERSION
    return record


def _remove_ball(view, atd, masks):
    """(ball_xy, ball_speed): the clean ball with each half's mask taken out of it, and the ball's speed NaN everywhere.

    `masks` is {half: (n,) bool}, True where the ball is missing, from the half's kick-off to the end of its graded
    block (_span). x and y go together, as an Atd requires; the rows outside the spans are untouched; our speed channel
    is NaN on 92% of the frames that have a ball (T31), so there is no ball speed to give. Shared by every profile that
    has a ball mask, so that their balls are made the same way. The arrays are read-only."""
    ball_xy = np.array(atd.ball_xy)
    for half, mask in masks.items():
        kickoff, stop = _span(view, half)
        ball_xy[kickoff:stop][mask] = np.nan
    ball_speed = np.full(view.n_frames, np.nan)
    for array in (ball_xy, ball_speed):
        array.setflags(write=False)
    return ball_xy, ball_speed


def _gap_ball(view, atd, calibration, seed):
    """(ball_xy, ball_speed, transplants, mask_share): the clean ball with our missing-ball masks transplanted onto it.

    Shared by the gaps profile and every profile that adds to it, so that their balls are the same bytes: the transplant
    draws are the "transplant" component's, whatever else a profile draws. The arrays are read-only."""
    masks, transplants, covered, missing = {}, {}, 0, 0
    for half, _, _ in view.graded:
        kickoff, stop = _span(view, half)
        choice = choose_transplant(view, half, calibration, seed)
        masks[half] = transplant(calibration.ball_masks[choice.source_half], stop - kickoff, choice.offset)
        transplants[half] = {"source_half": choice.source_half, "offset": choice.offset, "frames": stop - kickoff}
        covered += stop - kickoff
        missing += int(masks[half].sum())
    ball_xy, ball_speed = _remove_ball(view, atd, masks)
    return (ball_xy, ball_speed, {half: transplants[half] for half in sorted(transplants)},
            missing / covered if covered else None)


def _transplant_records(view, calibration, seed):
    """{half: {"source_half", "offset", "frames"}}: the Transplant of every half, as the cap reads it (cap_limits).

    The context profile transplants no mask, but its cap reads the counts of ours through the same draws as the
    tracking profile's, and the record says from which half of ours and which frame."""
    records = {}
    for half, _, _ in view.graded:
        kickoff, stop = _span(view, half)
        choice = choose_transplant(view, half, calibration, seed)
        records[half] = {"source_half": choice.source_half, "offset": choice.offset, "frames": stop - kickoff}
    return {half: records[half] for half in sorted(records)}


# -- the context profile's ball ---------------------------------------------------------------------------------------

def context_durations(calibration, fps):
    """[[int64 array] x 3] x 2: the Calibration's gap durations by onset class and third of the half, in frames at
    `fps`.

    A length is max(1, round(seconds * fps)), as the wrong-team episodes' are (calibrated_parameters); rounding is half
    to even. Each array is read-only, in the order the Calibration has them. ValueError, naming the cell, for a
    hazard_durations_s that lacks one of the six, or whose cell is not a list of lengths in seconds."""
    fps = _whole("fps", fps)
    table = []
    for onset_class in (0, 1):
        row = []
        for third in (0, 1, 2):
            try:
                seconds = np.asarray(calibration.hazard_durations_s[onset_class][third], dtype=np.float64)
            except (KeyError, IndexError, TypeError, ValueError):
                raise ValueError("the Calibration's hazard_durations_s has no usable cell [{}][{}] (gap lengths in "
                                 "seconds for an onset class and a third of the half); pass calibration.load()"
                                 .format(onset_class, third)) from None
            if seconds.ndim != 1 or seconds.size == 0 or not np.isfinite(seconds).all():
                raise ValueError("the Calibration's hazard_durations_s[{}][{}] must be a non-empty list of finite "
                                 "lengths in seconds; it has shape {}".format(onset_class, third, seconds.shape))
            frames = np.maximum(1, np.rint(seconds * fps)).astype(np.int64)
            frames.setflags(write=False)
            row.append(frames)
        table.append(row)
    return table


def context_probabilities(view, half, players_xy, calibration):
    """(prob, onset_class, third, defined, counts) of one half, each of the n rows from its kick-off to the end of its
    graded block; the arguments of hazard.sample_mask, and which rows have all their features.

    `players_xy` is the degraded players' (n_frames, N_SLOTS, 2), normalised. The features are hazard.features of the
    clean DFL ball and these players, in metres from a pitch corner on the view's pitch, with the half's kick-off as
    both block start and kick-off (the Calibration's spans). A row is `defined` where all six are: not where the ball is
    missing (DFL has none), nobody is present, or the ball's speed has no step to read yet. prob is
    hazard_c * HAZARD_SCALE * the model's probability there, at most MAX_HAZARD_PROB, and 0 elsewhere; onset_class is 1
    where the line distance is below calibration.NEAR_LINE_M; third is hazard.third_of_minute of the row's minute from
    the kick-off. counts: rows, ball_missing_rows (DFL has no ball), nan_feature_rows (a ball and a NaN feature) and
    capped_rows (a chance above the cap before it)."""
    kickoff, stop = _span(view, half)
    n = stop - kickoff
    pitch = np.asarray(view.pitch, dtype=np.float64)
    ball = np.asarray(view.ball_xy, dtype=np.float64)[kickoff:stop] * pitch
    players = np.asarray(players_xy, dtype=np.float64)[kickoff:stop] * pitch
    table = hazard.features(ball, players, view.pitch, view.fps, kickoff, kickoff)
    seen = ~np.isnan(ball).any(axis=1)
    defined = ~np.isnan(table).any(axis=1)
    raw = np.zeros(n)
    raw[defined] = calibration.hazard_c * HAZARD_SCALE * hazard.predict(calibration.hazard_model, table[defined])
    prob = np.minimum(raw, MAX_HAZARD_PROB)
    onset_class = np.zeros(n, dtype=np.int64)
    onset_class[defined] = table[defined, LINE_COLUMN] < calibration_module.NEAR_LINE_M
    third = hazard.third_of_minute(np.arange(n) / view.fps / 60.0)
    counts = {"rows": n, "ball_missing_rows": int((~seen).sum()), "nan_feature_rows": int((seen & ~defined).sum()),
              "capped_rows": int((raw > MAX_HAZARD_PROB).sum())}
    return prob, onset_class, third, defined, counts


def _mask_counts(mask, defined):
    """{"onsets", "missing_frames", "at_risk_rows", "post_gap_rows"} of one half's mask (True: missing).

    A gap is a maximal run of True and each is one onset (a gap cut by the half's end too). The at-risk rows are the
    rows that have a ball after the mask and all their features, except the half's last row, which no gap can follow.
    The post-gap rows are the at-risk rows within hazard.SPEED_STEPS frames of where a gap ends (its first row with a
    ball after it, and the rows after it): where a ball speed read from the degraded ball would still hold steps from
    before the gap, which the features of the clean ball do not."""
    first, stop = _runs(mask)
    at_risk = ~mask & defined
    at_risk[-1] = False
    after = np.zeros(mask.size, dtype=bool)
    for end in stop[stop < mask.size].tolist():
        after[end:end + hazard.SPEED_STEPS] = True
    return {"onsets": int(first.size), "missing_frames": int(mask.sum()), "at_risk_rows": int(at_risk.sum()),
            "post_gap_rows": int((at_risk & after).sum())}


def _context_ball(view, atd, players_xy, calibration, seed):
    """(ball_xy, ball_speed, record, mask_share): the clean ball with the hazard's gaps taken out of it.

    Each half's gaps are hazard.sample_mask on context_probabilities, from rng_for(seed, match_index, "hazard", half),
    with the durations of context_durations; the ball is made by _remove_ball, as the gaps profile's is. `record` is
    info["ball"]: the scale and c, and per half and in all the counts of the docstring of the module (no frame).
    `mask_share` is the share of the rows of the spans that the masks take: not the ball's missing share after them,
    which also holds the rows that DFL has no ball for."""
    durations = context_durations(calibration, view.fps)
    masks, halves = {}, {}
    for half, _, _ in view.graded:
        prob, onset_class, third, defined, counts = context_probabilities(view, half, players_xy, calibration)
        masks[half] = hazard.sample_mask(prob, onset_class, third, durations,
                                         rng_for(seed, view.match_index, "hazard", half))
        halves[half] = {**counts, **_mask_counts(masks[half], defined)}
    ball_xy, ball_speed = _remove_ball(view, atd, masks)
    totals = {name: sum(halves[half][name] for half in halves) for name in CONTEXT_COUNTS}
    record = {"scale": float(HAZARD_SCALE), "c": round(float(calibration.hazard_c), DECIMALS), **totals,
              "halves": {half: halves[half] for half in sorted(halves)}}
    return ball_xy, ball_speed, record, totals["missing_frames"] / totals["rows"]


# -- the offpitch profile's ball --------------------------------------------------------------------------------------

def line_distance_m(ball_xy, pitch):
    """(n,) float64, read-only: min(x, L - x, y, W - y) in metres of each row of a normalised ball; NaN where it is
    missing.

    `ball_xy` is (n, 2), normalised 0-1 as a TrackingView's, and `pitch` the match's own (length, width) in metres: x
    is scaled by the length and y by the width (the pitch is not square, and DFL's pitches are not all one size). The
    distance to the nearest touch or goal line, negative past it; a ball on a line is at 0, in play. The same quantity
    as hazard.features' line_distance_m and calibration.ball_stats', on the same metres (the normalised position times
    the pitch). A row with either coordinate NaN is NaN. ValueError, naming the argument, for a ball that is not (n, 2)
    and a pitch that is not two positive finite lengths.
    """
    xy = np.asarray(ball_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("line_distance_m: ball_xy must be (n, 2), the ball's normalised position on each row, got "
                         "shape {}".format(xy.shape))
    length, width = _pitch_m("line_distance_m", pitch)
    x, y = xy[:, 0] * length, xy[:, 1] * width
    line = np.minimum.reduce([x, length - x, y, width - y])
    line.setflags(write=False)
    return line


def offpitch_mask(line_m, margin_m):
    """(n,) bool, read-only: True on the frames that the off-pitch rule hides, with hysteresis.

    `line_m` is the line distance of each frame in metres (line_distance_m), NaN where there is no ball. A stretch is
    hidden from each frame with line_m < -margin_m (more than the margin past a line; exactly -margin_m is not past it)
    up to, not including, the next later frame with line_m >= 0 (back on the pitch; a ball on a line is), and to the end
    if none comes. Every other frame is as the frame before it: a ball between the line and the margin, or a frame with
    no ball (NaN, which neither starts nor ends a stretch), stays visible outside a stretch and hidden inside one. The
    first frame starts visible. Vectorised: the last frame back on the pitch and the last frame past the margin, each
    as a running maximum of its index, and hidden where the second is later. `margin_m` is at least 0 (an infinite
    margin hides nothing). ValueError, naming the argument, for a line_m that is not 1-D and a margin_m that is not a
    number of metres at least 0.
    """
    line = np.asarray(line_m, dtype=np.float64)
    if line.ndim != 1:
        raise ValueError("offpitch_mask: line_m must be 1-D, the line distance of each frame in metres, got shape {}"
                         .format(line.shape))
    margin = _margin("offpitch_mask", margin_m)
    index = np.arange(line.size)
    last_in = np.maximum.accumulate(np.where(line >= 0, index, -1))
    last_far = np.maximum.accumulate(np.where(line < -margin, index, -1))
    hidden = last_far > last_in
    hidden.setflags(write=False)
    return hidden


def thin_mask(mask, uniforms, drop_prob):
    """(n,) bool, read-only: `mask` with each of its interior gaps dropped whole where its uniform is below `drop_prob`.

    A gap is a maximal run of True (True: the ball is missing); an interior one touches neither index 0 nor n - 1, and
    an edge gap, which the span cuts, is never dropped. `uniforms` holds one number per interior gap, in time order (k
    of them: thinning_uniforms), and gap i is dropped (its frames set to False) iff uniforms[i] < drop_prob, so a
    drop_prob of 0 drops nothing and 1 every interior gap. The result is a fresh array and a subset of `mask`.
    ValueError, naming the argument, for a mask that is not a 1-D bool array, uniforms that are not one per interior
    gap, and a drop_prob that is not a probability.
    """
    flags = np.asarray(mask)
    if flags.dtype != np.bool_ or flags.ndim != 1:
        raise ValueError("thin_mask: mask must be a 1-D bool array (True where the ball is missing), it is {} with "
                         "shape {}".format(flags.dtype, flags.shape))
    first, stop = _runs(flags)
    interior = (first > 0) & (stop < flags.size)
    draws = np.asarray(uniforms, dtype=np.float64)
    if draws.shape != (int(interior.sum()),):
        raise ValueError("thin_mask: uniforms must hold one number per interior gap of the mask, {} of them, and has "
                         "shape {}".format(int(interior.sum()), draws.shape))
    dropped = draws < _drop_prob("thin_mask", drop_prob)
    edges = np.zeros(flags.size + 1, dtype=np.int64)
    edges[first[interior][dropped]] += 1                # the runs are disjoint, so no index is hit twice
    edges[stop[interior][dropped]] -= 1
    out = flags & (np.cumsum(edges[:-1]) == 0)
    out.setflags(write=False)
    return out


def thinning_uniforms(view, half, seed, n_gaps):
    """(n_gaps,) float64, read-only: the thinning's uniforms of one half, rng_for(seed, view.match_index, "thinning",
    half).random(n_gaps).

    One per interior gap of the half's transplanted mask, in time order (thin_mask). The fit of q (offpitch.py) reads
    the same uniforms for every q it tries, so that the profile and the fit agree on which gaps a q drops. ValueError
    for an n_gaps that is not a whole number of at least 0."""
    draws = rng_for(seed, view.match_index, "thinning", half).random(_whole("n_gaps", n_gaps))
    draws.setflags(write=False)
    return draws


def offpitch_masks(view, calibration, seed, margin_m, drop_prob):
    """({half: (n,) bool read-only}, record): the offpitch profile's ball masks, and what was drawn and hidden.

    On each half's span from its kick-off (_span, n = stop - kick-off rows), in this order: the transplant of
    tracking's draws (choose_transplant and transplant, exactly as _gap_ball reads them), thinned (thin_mask, with the
    half's thinning_uniforms and `drop_prob`), then ORed with offpitch_mask at `margin_m` of the line distance of the
    CLEAN DFL ball (view.ball_xy) in metres on view.pitch. True where the ball is missing. `record` is JSON-ready and
    holds no frame: margin_m, drop_prob, transplants ({half: {"source_half", "offset", "frames"}}), and per half
    ("halves") and in all the counts of OFFPITCH_COUNTS: the interior gaps of the transplanted mask, the gaps dropped,
    the frames they held (frames_restored), the off-pitch stretches and their frames, the final mask's frames
    (missing_frames) and the rows. ValueError, naming the argument, for a margin_m or a drop_prob out of range; as
    choose_transplant for a half the view has no block for and a Calibration with no mask.
    """
    margin, q = _margin("offpitch_masks", margin_m), _drop_prob("offpitch_masks", drop_prob)
    ball = np.asarray(view.ball_xy, dtype=np.float64)
    masks, transplants, halves = {}, {}, {}
    for half, _, _ in view.graded:
        kickoff, stop = _span(view, half)
        choice = choose_transplant(view, half, calibration, seed)
        base = transplant(calibration.ball_masks[choice.source_half], stop - kickoff, choice.offset)
        first, last = _runs(base)
        interior = int(((first > 0) & (last < base.size)).sum())
        thinned = thin_mask(base, thinning_uniforms(view, half, seed, interior), q)
        off = offpitch_mask(line_distance_m(ball[kickoff:stop], view.pitch), margin)
        mask = thinned | off
        mask.setflags(write=False)
        masks[half] = mask
        transplants[half] = {"source_half": choice.source_half, "offset": choice.offset, "frames": stop - kickoff}
        halves[half] = {"interior_gaps": interior, "gaps_dropped": int(first.size - _runs(thinned)[0].size),
                        "frames_restored": int(base.sum()) - int(thinned.sum()),
                        "offpitch_stretches": int(_runs(off)[0].size), "offpitch_frames": int(off.sum()),
                        "missing_frames": int(mask.sum()), "rows": stop - kickoff}
    totals = {name: sum(halves[half][name] for half in halves) for name in OFFPITCH_COUNTS}
    record = {"margin_m": margin, "drop_prob": q,
              "transplants": {half: transplants[half] for half in sorted(transplants)}, **totals,
              "halves": {half: halves[half] for half in sorted(halves)}}
    return masks, record


def _offpitch_constants():
    """(OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB), read when the profile is made; ValueError while any of the two, or the
    fit's record OFFPITCH_FIT, is None (not fitted yet)."""
    unfitted = [name for name, value in (("OFFPITCH_MARGIN_M", OFFPITCH_MARGIN_M),
                                         ("OFFPITCH_DROP_PROB", OFFPITCH_DROP_PROB), ("OFFPITCH_FIT", OFFPITCH_FIT))
                if value is None]
    if unfitted:
        raise ValueError("profile {!r} is not fitted yet: degrade.{} {} None. Its margin and drop probability are "
                         "fitted once, on label-free statistics, by `python pipeline/benchmark/offpitch.py fit` "
                         "(R015), and pinned in degrade.py with the fit's record; until then the profile cannot be "
                         "made. Use the {} profile (the default)".format(
                             OFFPITCH, " and degrade.".join(unfitted), "is" if len(unfitted) == 1 else "are",
                             TRACKING))
    return OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB


def _margin(function, value):
    """`value` as a float of metres, at least 0 (inf allowed: nothing is past it); ValueError naming the function and
    margin_m otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not value >= 0:
        raise ValueError("{}: margin_m must be a number of metres, at least 0 (the margin past a line beyond which the "
                         "ball is hidden), got {!r}".format(function, value))
    return float(value)


def _drop_prob(function, value):
    """`value` as a float in [0, 1]; ValueError naming the function and drop_prob otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not 0 <= value <= 1:
        raise ValueError("{}: drop_prob must be a probability, at least 0 and at most 1 (the chance that an interior "
                         "gap is dropped), got {!r}".format(function, value))
    return float(value)


def _pitch_m(function, pitch):
    """(length, width) as two positive finite floats; ValueError naming the function and pitch otherwise."""
    try:
        length, width = (float(side) for side in pitch)
    except (TypeError, ValueError):
        raise ValueError("{}: pitch must be (length, width) in metres, got {!r}".format(function, pitch)) from None
    if not (math.isfinite(length) and math.isfinite(width) and length > 0 and width > 0):
        raise ValueError("{}: pitch must be two positive finite lengths in metres, got {!r}".format(function, pitch))
    return length, width
