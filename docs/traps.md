# Traps

Things that cost us time or would cost us points. Add one whenever something takes more than half an hour to figure
out: what you saw, why it happens, what to do instead. Give it the next free ID and put it in the right section.

## Scoring

**T01 — No tuning on the public window: a rule, not just a bad idea.** Since the organisers' clarification of 23 Oct
2026 (upstream `f8f2629`; RULES.md rule 2, EVALUATION.md) the public 5 minutes are for checking that the pipeline is
correct (frames aligned, file well-formed, labels meaning what you think), and they are "not training data and not a
tuning set". Choosing between two versions on it, setting or calibrating a constant on it and training on it all break
that rule, out of fold or not. It would also pick the wrong answer. The window holds 17 possessions, so one possession
is worth about 0.06 spellF1. It is the best-covered 5 minutes of the match: the ball is missing on 12% of its frames,
against 40.7% overall and up to 80% per 5-minute block
([A02](results.md#a02--ball-coverage-across-the-match-2026-10-21)). And it ranks `smooth125` above `smooth50`, the
reverse of the full match. Use it to check correctness only, and decide on P01 validation: per-match t-intervals and
the exact sign-flip test of [`pipeline/benchmark/README.md`](../pipeline/benchmark/README.md), never a bootstrap.
The organisers' clarification of 3 Nov 2026 ([rules.md](rules.md)) keeps this: no "training, fitting or calibrating
your algorithm on the ground-truth possession data", while "running automatic checks using only the provided data" is
allowed. Scoring a file on the window is such a check; it may stop a candidate and never ranks two.

**T02 — `X` means "repeat my last label" — and at the start of a half, "copy my first label".** Before computing
spellF1, the scorer forward-fills `X` with the last non-`X` label in the half, and back-fills the head of the half.
- The second half has 2,934 graded pre-kick-off frames (110,808–113,741), all of them `D`. An `X` across them becomes
  whatever we say at kick-off: a long spurious possession.
- An `X` after the ball goes out extends the possession into the stoppage.

Write `D` in both cases.

**T03 — Latency is expensive.** Perfect labels 1 s late score spellF1 0.65, and 8 frames late score 0.89
(EVALUATION.md). The shortest public possessions tolerate only 0.3–1.5 s of total boundary error at IoU 0.7 (A01).
Hysteresis and "k frames of control" add lag unless the change is back-dated to where the evidence starts. Check
`lat_f` on the board and the latencies in `evaluation/reports/<name>_transitions.csv`.

**T04 — A blip costs three possessions.** Five frames of `A` inside an `H` possession split it in two and add a
spurious `A`; a short `D` blip splits it too. But real short possessions exist — 0.76 s and 1.32 s in the public GT —
so minimum-duration filters need validating, not guessing.

**T05 — `H H D D H H` is two possessions.** Missing a stoppage merges the possessions on either side of it, and
inventing one splits a possession.

**T06 — Possession counts differ between tools.** `validate_submission.py` treats `X` as a gap and counts over the
whole file. The scorer bridges `X` and counts inside the graded halves only. Compare like with like: A01 counts the
baselines over the graded halves.

**T07 — Noise floors.** On the full match, spellF1 gaps below ~0.04 and macroF1 gaps below ~0.01 are not real. On the
public window, one possession is worth ~0.06.

**T33 — DFL's possessions are fine-grained, and the public window cannot say by how much ours differ.** DFL's flag gives
437 possessions a match, a median of 3.7 s (the literature's 175–220 is for control-based rules, 420 for every touch).
It is tempting to read the public window's 4.23 possessions per alive minute against DFL's 8.30 as "ours is half as
fine", and it is not safe: the window is alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows, and
DFL's rate falls as the alive share rises (r = −0.71). Against DFL's windows that are as lively the window is 1.34 times
slower, and a quarter of them are as slow:
[A06](results.md#a06--dfls-conventions-against-the-public-window-and-the-literature-2026-10-23).
So the grain of our GT is not known, and neither is what a lag, a smoothing level or a switch penalty costs on it:
EVALUATION.md's lag table (0.9818, 0.8933, 0.6539 at 3, 8 and 25 frames) does not reproduce on the public GT (1.0000,
0.9529, 0.7765) and does not say what it was computed on. Rank every choice that depends on possession length under the
raw DFL truth and under the coarse one (`truth.coarse_codes`, `COARSE_MIN_S` = 5 s) side by side, and treat a
disagreement as a warning. The coarse truth is a sensitivity check, not a correction. Pick hyperparameters on the
benchmark, never on the public window (T01).

**T40 — Our ground truth's dead-ball edges and team flips need not follow DFL's flag, so a component tuned on the
benchmark may carry DFL's timing.** The benchmark's truth is DFL's flag, and P04's back-dating, P05's model and P06's
decoder were chosen against it. A11 timed the changes of our public truth against label-free events on our tracking and
set them beside DFL's (profile `tracking` v3, on its degraded copy; the numbers are the raw truth's, and the coarse
truth's are given where they differ). Our `D` begins about 1.5 s before the ball is seen beyond a line (DFL's flag, both
truths: 0.12 s after), and our `D` ends 0.36 s before the restart's first controlled touch and about 1.5 s after the
model's alive crosses 0.5 (DFL's flag: 1.40 s before the touch, 1.48 s coarse, and at the crossing under both). Our team
changes come 0.96 s before the new team's first controlled touch (DFL's: 0.04 s after it, and on it under the coarse
truth). Three of those gaps have a 90% interval that excludes 0 under both truths (`line_out`, `touch_start` after a
restart and `model_alive`); `model_dead`'s and the team gap's do not. But each rests on 2 to 4 public changes found (the
team cell on 10 of 12), and the detectors run on our real tracking against a degraded copy of DFL's. Convention is one
reading; tracker coverage predicts all four `D`-edge signs by itself (our tracker loses the ball near the lines, T39,
and the copy's ball gaps are blind to the play, R003), and so does chance at this n. A11 does not separate them. It is a
lead and not a measurement. What to do:
- do not read a timing parameter off DFL as ours (a back-dating, a `D` onset, a threshold on the model's alive): the
  model and DFL's flag agree to a frame out of fold (+1 and 0 frames), and that says nothing about our truth;
- before shifting any edge, or adding a feature that places one, read A11 for where our truth looks different and R012
  for what our pipeline's own latency is (on DFL and at those 20 changes) and which shift the rule adopted (the
  pipeline's own lag on DFL, 1 to 2 frames, and no gap to our convention), and shift only by the rule declared for
  `shift_edges`, with its public half retired (below);
- a person can watch the 20 changes on the tactical video (`conventions.py public`, A11's "How a person can extend
  it") to understand whether `D` begins at the line, the whistle or the stoppage, and what the flip of a team change
  is, and what they learn may shape a feature. It never becomes a number in a config, a label or a rule tied to those
  frames (T27): a constant that fixes our convention would be tuned on the public window. Since 3 Nov 2026
  ([rules.md](rules.md)) watching is only "to better understand what counts as possession", and "checking your output
  against the video" is not allowed: the person watches the truth's changes and ignores the file's detector columns,
  which are our own components' events.

*Changed 23 Oct 2026 (`f8f2629`):* R012's rule had two parts, s_b (the pipeline's own lag on the seven DFL matches,
allowed) and g (the gap to our convention, read from the public window's latencies, which is now a tuning use of the
window and not allowed). g came out 0 for every type, so `main.toml`'s shift is s_b alone, from DFL. g is **retired**:
it stays 0 by rule, `conventions.py shift` still prints it for reference, and any future shift comes from other games
only (P16).

## Alignment

**T08 — Off by one.** ATD `frame_id` runs 1…199,319, while submission rows, homography `frame` and video frames all
start at 0. Smart Tagging uses seconds: frame = `round(s * 25)`.

**T09 — ~164k rows means you indexed the broadcast.** The broadcast has its own timeline and no mapping to the
tactical one.

**T10 — The `offsets` in `games.py` are not for the ATD.** They map to the internal IdFrame counter the GT was derived
from.

**T29 — Two answers for each kick-off frame.** DATA.md, FAQ.md and `starter/load_smart_tagging.py` put the kick-offs
at frames 11,185 and 113,742. The `(minute, second, frame)` anchors in `games.py` convert, through
`games.anchor_video_frame`, to 11,186 and 113,741: one frame off, in opposite directions. Use the documented pair,
which the graded windows agree with (the first half is graded from its kick-off). The pipeline exposes it as
`inputs.kickoffs`.

**T11 — Temporal code must run per half.** The baseline's majority vote ignores half boundaries; ours must not. Use
`games.half_video_blocks("BAR-ATH")`, which gives 11,185–90,651 and 110,808–190,197. The first half is graded from its
kick-off frame; the second is graded from 2,934 frames before its kick-off at 113,742.

## Data

**T12 — A missing ball is not a dead ball, in either direction.** In the public window the ball is visible on 66% of
dead-ball frames (placed, held, thrown in), and over the match, in-play gaps of 10 s or more are normal (A01).

**T13 — Ball gaps are long, not flickers.** There are 150 gaps over the graded halves: median 10.7 s, p90 34 s,
max 3.7 min. Gaps under 2 s hold 0.1% of the missing frames, so interpolation buys almost nothing (A01). Coverage also
gets worse later in each half (A02).

**T14 — The nearest player is not the possessor.** The nearest player's team is wrong on 22% of in-play,
ball-visible public frames (A01): passes in flight, duels, deflections.

**T15 — Track slots are tracklets, not players.** There are 46 slots for 22 players; they appear, vanish, swap, and
occasionally duplicate someone (more than 22 objects). `ShirtNumber` is just the slot index. Build collective,
permutation-invariant features. A slot does carry the wrong team at times, after an identity switch: a team shows more
than 11 slots while the other shows fewer on 10.8% of graded frames (T32, A07).

**T16 — Units and orientation.** Coordinates are normalised 0–1, and the pitch is not square.
- Scale by 105.3 × 68.0 m (`games.py`; the metadata says 105 × 68) before computing distances.
- Feed the homography normalised coordinates, not metres.
- Teams swap ends at halftime, so derive each team's attacking direction per half before building directional
  features.

**T17 — Smart Tagging quirks.**
- Codes are upper case, values are title case, and kloppy's tag columns are title case, so match case-insensitively.
- The away team is spelled `Atletic Club`.
- 36 tags have Team `N/A`, including every throw-in.
- Windows open seconds early (kick-off tags by ~3.5 s), and phases are not possessions. A `SET PIECES` tag is a fixed
  875-frame (35 s) clip, a kick-off tag 125 frames (T55).
- `DEFENDING …` and `DEFENSIVE TRANSITION` name the team *without* the ball.

**T18 — The camera lags the ball.** The tactical camera's aim follows the ball's along-pitch position about 35 frames
late, with a median error of 12 m, and carries no information about the across-pitch position (A03).

**T28 — Speed is not presence.** The ATD's speed channel is NaN on 370,297 slot-frames and 103,579 ball frames where
the position exists (measured 2026-10-22). Test presence on x, which is always missing together with y; the
pipeline's `atd.players_present` and `atd.ball_present` do exactly that.

**T19 — kloppy is slow on the full ATD.** Loading all 199,319 frames takes minutes and gigabytes. Stream the raw file
once, as `starter/baseline.py` does, and cache arrays under `data/cache/`.

**T30 — kloppy's DFL loader hides the dead ball.** Both of kloppy's DFL loaders,
`kloppy.sportec.load_open_tracking_data` and `kloppy.sportec.load_tracking`, default to `only_alive=True` and silently
drop every dead-ball frame. That leaves no `D` to learn or score. Pass `only_alive=False`.

Measured on the seven DFL matches on 2026-10-22 (kloppy 3.18.0; `python pipeline/benchmark/selftest.py --dfl`):
- **The default drops the dead ball.** On J03WMX, `only_alive=True, limit=5000` returned 5,000 frames over N 10,000 to
  15,823: none is dead in our parse, no alive row in that range is missing, and the 824 dead rows are the ones it
  dropped. Dead frames are 40.7% to 51.7% of the ball frames of a match (J03WQQ the least, J03WN1 the most), so the
  default deletes about half of every match.
- **`limit` is per object and counts what the loader keeps.** With `only_alive=False` it stops after the first 5,000
  frames of the first half (N 10,000 to 14,999, period 1 only). With the default it counts alive frames, so the same
  limit reaches further.
- **With `only_alive=False` it agrees with our parser** on those 5,000 frames: ball x, y and z, alive or dead, the
  team in possession and three players, with 0 disagreements. Our frame numbers and its `frame_id` are the same N.
- **The other two behaviours never fire on these files.** The loader reads `BallPossession` as home when it is 1 and
  away otherwise, so a frame with no possession value would come out as away, and it skips a frame with no ball
  entry. In all seven matches every ball frame has a `BallPossession` of 1 or 2 and a `BallStatus`, and no frame
  number in a half is without a ball frame (0 in each count). Our own parser (`pipeline/benchmark/dfl.py`) counts the
  first two and refuses any other code, so a file that changes this stops the parse.

Its own download does not work: kloppy 3.18's DFL loader fetches from `springernature.figshare.com`, which sits behind a
bot challenge, so `load_open_tracking_data` cannot get the files. P01 downloads them with
`pipeline/benchmark/sources.py` and parses them with `pipeline/benchmark/dfl.py`. kloppy still reads local files
(`kloppy.sportec.load_tracking(meta_data=..., raw_data=...)`), which is how the cross-check runs.

If you do use kloppy on these files, index frames by `frame_id` rather than by position, and read `BallPossession`
yourself for the away team.

**T31 — The ATD ball-speed channel is unusable.** Where the ball's position exists, its speed is NaN on 92.2% of the
graded frames (measured again on 2026-10-23: 92.17% of the 94,214 ball-present graded frames, 91.97% of those from each
half's kick-off, A07). Where it is given, its median is 1.3 m/s against 4.8 m/s for the ball's speed from its own
positions, and its correlation with that speed is at most 0.11 at any lag (the 2026-10-22 measurement, P01's plan): it
behaves like a player's speed, not the ball's. Compute the ball's speed from successive positions (`hazard.features`
does) and never read `atd.ball_speed`: a feature built on it is missing on nine frames in ten and noise on the tenth.
The benchmark's degraded tracking sets `ball_speed` to NaN everywhere for this reason. T28 says the speed channel
is not presence; this says it is not the ball's speed either.

**T32 — A slot can carry the other team's player.** After an identity switch a slot keeps its team in the metadata while
it follows a player of the other side, so a team shows more than 11 slots while the other shows fewer than 11, which a
real match cannot do. Measured on the graded frames (A07): this happens on 10.8% of them (17,156 of 158,857), in 108
episodes (63 with the home team in surplus, 45 the away team), median 2.7 s, p90 16 s, longest 50 s; from each half's
kick-off, the span that the calibration stores, it is 10.5% of the frames (16,433 of 155,923) in 104 episodes. A team
shows more than 11 slots on 12.9% of frames, and on 84% of those the other shows fewer than 11 (the 2026-10-22
measurement, P01's plan). So `atd.slot_team` is the label of the slot and not of the player on it, and a feature that
trusts one slot's team (the nearest player's, say) is wrong on those frames. The 10.8% is a lower bound: a swap inside
an 11 against 11 cannot be seen. P01's `tracking` profile puts such episodes into the degraded matches, and is checked
against this share (R004).

**T46 — The ball is about 6 px wide in the tactical video, smaller than a vision transformer's patch.** A14 measured
4.7 to 8.2 px (p10 to p90, median 6.0) on the frames where the ATD has the ball, at about 30 px per metre. A ViT or
DETR-family detector (DINOv2, RF-DETR, OWLv2, Grounding DINO) cuts the image into 14 to 16 px patches, so the ball is a
fraction of one token. Any model that shrinks the frame first (TrackNet and WASB's usual 288 × 512 input, a COCO detector
at 640 px) turns it into 1 or 2 px. Ball detection on this video needs full resolution, or tiles of it, and a
small-object architecture: a stride-4 heatmap net, with stacked frames for motion. Check the ball's pixel size before
choosing any vision model. The homography gives it at every pitch point (A14).

**T35 — DFL's jitter is a few jumps on a 1 cm grid, and what the benchmark first blamed on them was our own slot
handovers.** The RMS of the second difference of a DFL player's position (what `calibration.player_stats` calls jitter)
is 9.0 cm over the 7 matches, against 3.3 mm on ours, which is what a rule of the kind "smooth if more than twice ours"
reads, and it is not noise. In the five 2. Bundesliga matches 98% of its squares sit in the largest 0.05% of the
triples: rare jumps (in one half 61 steps of 1.52 million exceed 25 m/s, and the largest is 30 m). The rest is
quantisation: the median second difference is 1.00 cm in every match and the 99th percentile 2.83 cm, against 1.4 mm
and 9.6 mm on ours. That much is DFL's. **The 6.5 to 8.5 cm that the `tracking` profile still showed after its 5-frame
average in versions 1 and 2 was not**: the slot rule let a tracklet take a slot on the very row it was freed, so one
slot held a player up to row t - 1 and another from row t: in version 2 372 times in 7 matches x 3 seeds (17.7 a match
and seed), the two players 13.8 m apart at the median and 40.1 m at the 90th percentile, in version 1 236 times, 5.3
and 43.0 m. Ours has none: 3.28 mm over 3.2 million triples leaves no room for one, and a single jump of 12 m adds 8
times the squares of all of ours. On the same draws, measured per slot and then only inside tracklets, version 2 gave
8.49 and 1.75 cm after the average (version 1: 6.47 and 1.80 cm), and the 744 triples that straddle a handover held 96%
of the squares per slot. The reading of R004 and R004b, that the average removes little because DFL's jumps stay in,
was a reading of the handovers (R004c). Version 3 frees a slot only from the row after its tracklet, so a slot never
changes player between two adjacent rows and the jitter per slot is the jitter inside tracklets: 8.90 cm without the
average (DFL's own 9.00 again) and 1.80 cm with it, three seeds (8.75 and 1.77 cm at seed 0). Inside a piece the
centred 5-frame average takes the RMS down to a fifth, which is the arithmetic of a jump: a step of J between two
frames is two triples of +-J, and after the average two of +-J/5 (white noise falls to 16%). The 1.8 cm that is left,
5.5 times ours, is DFL's, lowered; whether it is the jumps or the grid inside a piece was not measured. **No jump
filter is needed**: the profile's own average takes the jumps down by that factor, and the only jumps left in the slot
trajectories are DFL's own. So when a method reads positions, speeds or accelerations, compare medians and percentiles
and not an RMS, and know that the benchmark's players are rougher than ours by 5.5 times in RMS
(`test_data_tracking_constants_follow_from_the_measurements` prints it; R004c).

**T36 — Two DFL matches have a team down to 10, and a gate on how many players are present cannot see it.** J03WN1's
away team has 10 players from minute 7.1 of the first half to the end of the match (92.5% of its graded frames) and
J03WQQ's from minute 13.4 of the second half (37.2%): 18.3% of the seven matches' graded frames have 21 players, and
none has more than 22. A statistic about how many slots are present (the share of frames with all 22, the mean) on all
frames is pulled down by the missing player whatever the degradation does, while ours is one match that is assumed to
have 22 players on the pitch throughout. The first version of R004 gated on all frames and passed on two effects that
cancel (a profile with too many frames at 22, and 18.3% of frames that cannot have 22); with the tail it would have
failed for a reason that had nothing to do with it. Measure such counts on the frames where the clean match has 22
players (`fidelity.COUNT_GATES`, `fidelity.full_strength_blocks`), and look at the frames where a statistic cannot reach
its value before trusting it. The wrong-team episodes are about half unobservable on 10 against 11, which lowers that
share in those two matches (R004).

**T37 — The walk's tracked spans are ours *after* the cuts, so every part of the profile that cuts counts them twice.**
The `tracking` profile draws a tracked span from our tracklet lengths, which are our FINAL tracklets: they already end
where our tracker loses a player, where an identity swaps and where the camera leaves players out. It then cuts the
spans again, with the wrong-team episodes and with the cap of our low-count frames, and the benchmark's tracklets come
out too short: pooled over the seven matches and three seeds the walk alone gives a median of 40.88 s against ours 41.28
s, the episodes alone 29.26 s, the cap alone 24.44 s, both 19.68 s (R004b, with version 2's slot rule), and the first
version of the cap failed the tracklet-median gate for it. A piece that a cut leaves under 8 frames is also no tracklet
of ours (A09: the shortest is 8 frames), and version 2 without the rule made 2.2% of its pieces so (906 of 40,866; 195
of them already short before the cap), because our count crosses the threshold 18 on 155 frames, 147 of them by one slot
(18 and 19), and the dropped players come back for a few frames and go again. `degrade.TRACKED_SCALE` (2.072, the walk's
drawn lengths multiplied once, from one label-free measurement, frozen; 2.042 until version 3's slot rule moved the
measurement, R004c) and `degrade.MIN_PIECE_FRAMES` (8) are the corrections. Any later part that cuts tracklets in the
benchmark must be set against the tracklet median first, and a rescale of this kind is made once, logged with its
measurement and never tuned.

**T38 — The default profile cannot value evidence about where the ball's gaps fall, so a choice that depends on them is
read on `context` too.** `tracking` v3 transplants our ball masks with their pattern and not their clock (R003), so a
gap has nothing to do with the play. In our public window the ball is missing on 33.6% of dead frames against 6.5% of
in-play frames (A01), a missing frame being 5.2 times likelier in a stoppage; on `tracking` it is visible on 56.8% of
the dead frames against about 59% of all graded frames (R005), so a missing frame says almost nothing about `D`. P06's
`ball_presence` is evidence of exactly that kind, so R006 and R007 chose `switch_dead` on a profile that cannot value
it, and A10's `__alive` ceiling (+0.1625 raw) starts from almost no `D` information and overstates what a perfect
detector would add on our match. `context` places the gaps by a hazard fitted on our tracker (46.2% of its dead frames
are visible, against 56.9% of all frames, R005), and the chosen config's position against the yardsticks improves on it,
as the design predicted (R007). It fails two gates and is a sensitivity profile, so it is a second look and not a second
validation. Before adopting a source or a parameter that depends on the gaps (`ball_presence`'s `margin`, `switch_dead`,
P05's and P07's features, P09's camera prior), run it on `tracking` and on `context`, under both truths, and treat a
disagreement among the four cells as a warning (as T33 does for the truths); whether `context` should veto is for the
team to decide. Say which profile each number is from. The `margin` follow-up in `proposals.md` ("After P06") proposes
one such rule, for `margin`.

**T39 — DFL tracks the ball out of play, so a dead-ball model looks near-perfect on the benchmark, and the benchmark
cannot say how much of that transfers.** On `tracking` v3 P05's boosted trees reach an alive/dead F1 of 0.9725 (the
yardsticks' 0.6094 and 0.6428) and 95.2% of A10's perfect-`D` ceiling under the raw truth (R009). They split first on
where the ball is: beyond a line, next or last seen beyond a line, at a restart spot. DFL's tracker keeps the ball while
it is out of play and ours loses it near the lines, so those features are visible during DFL's stoppages far more often
than during ours: among the rows where a gap could begin, the ball is beyond a line on 17.68% of DFL's against 4.07% of
ours, and within 2 m of a line on 25.23% against 9.01% (R009, A08). That is where the visible ball is, and not how often
it is missing, which is why `context` (T38) does not remove it: it places the gaps by our hazard but keeps DFL's ball
out of play. Reading numbers for any source that uses the ball's position near the lines (the dead ball, a restart
spot, a throw-in):
- call the benchmark's score an upper bound for our match, whatever the profile, and say so beside the number;
- look at the label-free signs on our match, the predicted `D` share (36.0% against 44.6% on the benchmark, R010) and
  the public window's alive agreement (86.6% of 7,500 frames), as aggregates and not as a score;
- do not choose a decoder parameter that trades `D` against the team (`switch_dead`, a source's weight) from that
  source's benchmark alive/dead score alone.

R015 hid DFL's out-of-play ball (`offpitch` v1) and today's P05 lost 0.013 alive/dead F1 (0.9731 to 0.9600): a lead, since
the profile failed a gate (T45), that the overstatement through the visible out-of-play ball is small.

**T45 — A degradation fitted to one statistic can move the others away, and a cheap feasibility sweep finds that first.**
R015's `offpitch` v1 matched the share of visible ball frames past a line within its gate (0.0414 against 0.0407), failed the median
gap (15.04 s against 10.74, +-1.5) and moved two checks away (the ball visible on dead frames, 0.3667 against 0.6637; the
near-line onset share, 0.5400 against 0.2797). Its v2 mechanism could not reach the share at all (0.258 to 0.262 at every
margin), which a label-free sweep of nine minutes showed before anything was built (R015b). Before building a profile
around a statistic:
- sweep its parameter on all the gate statistics with a throwaway label-free script, and read the whole table, not the
  fitted statistic alone;
- declare which statistics are fitted and which are checks, as R015 did for the share and the near-line onsets;
- a mechanism designed after seeing a profile's gates makes those gates less independent of it, so say so beside the
  result.

**T47 — A noise model fitted to a residual's variance and lag-1 autocorrelation can still get the speed wrong.**
R019's simulated camera followed the DFL ball plus AR(1) noise fitted to our camera's residual (phi near 1). The fit
was made to those two numbers, but with phi that near 1 the innovation is the residual's frame-to-frame change, with a
median step of 0.67 sigma a frame (1.14 m/s here). A camera that mostly stands still was simulated as one that never
does: 1.95 m/s in play and 1.12 m/s dead, against ours 0.30 and 0.04. Fit, or gate, the statistic the feature will read
(here each state's median speed of the aim); R019's gate 4, a dead/alive ratio, caught it only indirectly. Cost: P09's
simulator built and then dropped.

**T43 — An oracle that stops where the truth changes measures more than a source can deliver.** A12 sized P17's
prize with a perfect 2 s vote for the restarting team (+0.0198 raw spellF1), and that vote stopped at the truth's next
change of team. A real vote has a fixed length and cannot know that change: it keeps pushing after the restarting team's
first possession has ended, and DFL's possessions are short (median 3.7 s). R014 delivered +0.0040 of the +0.0198 (and
raw spellF1 fell with every longer tail, to -0.0187 at 150 frames). Before taking an oracle's gap as a prize:
- build the oracle with the real source's footprint (a fixed span, no truncation at the truth's changes, no knowledge of
  where the truth's next possession ends), and read that gap as the ceiling;
- keep a perfect-information oracle as an upper bound only, and say in the protocol how far below it a source is
  expected to land.

**T44 — A transition price binds only below the largest evidence it competes with.** R014b's `switch_restart` grid
{4, 8, 16, inf} could not differ: the restart evidence per stoppage is one observation, at most logit(0.93), about 2.6
nats (a throw-in 2.44), so choosing `D_H` or `D_A` for a `D` run moves its score by at most that, while a cross
transition costs `switch_restart` on top. Any value above about 2.6 decodes like `inf`; the identical labels across the
grid were guaranteed by the grid, not observed. Before declaring a penalty grid, compute the largest evidence the
penalty competes with and put the grid's values on both sides of it; a value above it is a duplicate of `inf`.

**T50 — A decoder's posterior is a product of per-frame terms, so it is several times too sharp, and a rule that reads
it barely moves; tempering it does not say which way to move.** The `hmm` adds one emission a frame over frames that are
strongly correlated, so its marginals count the same evidence many times. A19 measured them four to five times too
sharp (Platt α 0.27 for alive, 0.19 for the team), while each source alone is near calibrated (`dead_ball_model` 1.00,
P07 1.07 to 1.10). R025 moved each change to the posterior's median, which sat within a frame or so of Viterbi's: 55%
of the changes moved, by a median of 1 frame, and the score did not move (mean of the two truths -0.0010, 2 wins in 7).
R025b, the one retry, tempered the posterior at T = 5 from A19's α: the edges moved by a median of 3 frames, but frames
of weak evidence are as often the old possession's tail as the new one's start, so the edges' errors widened and the
score fell (-0.0282, 0 wins in 7). Cost: two declared experiments, their code and their benchmark runs, then the code
removed (recoverable from `171d0bb`). R025's gate was on the prize (A19's clean-pair ladder loss, 0.0704), and A19 had
already shown the posterior too sharp for the mechanism. Before building on a decoder's posterior:
- measure its calibration first (`pipeline/benchmark/diagnose.py`), and gate on the mechanism as well as on the prize;
- do not read `posterior = true`'s array as a probability: where the evidence is not flat, a threshold or a median on
  it falls on Viterbi's own frame;
- a flatter posterior spreads an edge without saying where the truth is; what is left of the edges' loss is where the
  evidence changes, which a better source, not a rule on the posterior, has to fix.

**T53 — A stand-in copied from our match fails on DFL wherever DFL's served world differs from ours, so measure that
difference before building the stand-in.**
- **R028's stand-in** was a simulated broadcast. It transplanted BAR-ATH's real fill masks gap by gap, by the gap's class
  (alive or `D` under the decoder) and its length.
- **The code was right.** A world made of the donors' own gaps gave back our placement statistics exactly: F1 to F4,
  W1 and W2, with F5 within 0.02 m (the advisor's identity test).
- **DFL's world was not ours.** On DFL, `tracking`'s gaps fall blind to the play (T38): under the driver, 54% of their
  frames are alive, against 20% of ours (measured). Their lengths by class differ too. We read this as the reason the
  per-frame rates came out unlike ours: fill on alive frames 0.218 against 0.295, on `D` frames 0.089 against 0.041.
- **The cost.** The lane was dropped at its first gate, 12 minutes into its runs, after a day of building. R019's
  simulated camera failed its fidelity gates the same way.

What to do:
- before a stand-in's wiring is built, compute what its gates depend on directly on DFL's served world, beside ours.
  Here that was the share of gap frames the decoder calls alive (W2) and the gap lengths per class. It needs one decode
  and no stand-in at all;
- if those already differ by more than the gates allow, the transplant cannot pass without being tuned towards the gate,
  which T26 and T37 forbid, so change the design first. For R028, gaps placed by the play (P01's `context` profile, a
  sensitivity check only, R005) differed as well: W2 0.44.

Cost: about a day.

## Reproducibility and environment

**T20 — Use the team venv.** It runs Python 3.11.9 in `.venv`, with numpy 2.4.6, pandas 3.0.6 and kloppy 3.18.0.
System Pythons differ — a machine's default may be 3.12 with numpy 2.5 — and different library versions can change
floats, sort orders, and therefore the file. Pin the final versions in the submission repo.

**T21 — Line endings change the file.** `open(path, "w")` writes `\r\n` on Windows and `\n` elsewhere, so the same
code produces different bytes on different machines — and "the same file" may be checked byte for byte. Write with
`newline="\n"` (in pandas, `lineterminator="\n"`). Git's `core.autocrlf` also rewrites CSVs on checkout, so compare
hashes of freshly regenerated files, not of the working tree.

**T22 — Nondeterminism.** Seed everything. Set deterministic flags for GBDT and PyTorch (thread counts, GPU kernels),
or run final inference on CPU. Sort directory listings.

**T23 — Video decoding.** A decode must yield exactly 199,319 frames; don't pad or trim (FAQ). OpenCV's
`CAP_PROP_POS_FRAMES` seeking is not frame-accurate, so decode sequentially or verify the indices.

**T24 — The scorer reads every CSV in `evaluation/results/`.**
- Stale files clutter the board, so use `--results <dir>`.
- Don't name our files `baseline_*.csv`: the gitignore exception for that pattern would commit them.
- `python starter/baseline.py` without `--out` overwrites the committed baselines.

**T34 — `np.load(path)` can leave its file open, and Windows then refuses to rewrite it.** A zip cut short that carries
an `.npy` name fails inside numpy's `NpzFile`, after `np.load` has handed the file over, so nothing closes it (a
`ResourceWarning` on Linux and macOS; on Windows the rebuild that follows an unreadable cache, which rewrites the file,
dies with `PermissionError` while the exception is still alive). Open the file yourself and give `np.load` the handle
(`with open(path, "rb") as handle: np.load(handle, allow_pickle=False)`), as `dfl._read_array` does.
`calibration._read_array` does the same now: it opens the file, and a zip cut short under an `.npy` name is a cache
that is rebuilt, not a `BadZipFile` (`test_calibration_cache_is_keyed_and_unreadable_means_absent` damages one).
`pipeline/inputs/atd_cache.py` still passes a path to `np.load` and does not catch `zipfile.BadZipFile`, so there a zip
cut short under an `.npy` name raises instead of rebuilding (nothing makes such a file; a note for whoever next edits
it).

**T41 — Hours of benchmark runs can take a machine down, and a grid's process writes its reports only at the end.**
During R013 one team machine (Windows, 12 logical cores, 15.3 GB) crashed with bugcheck 0x133 (DPC_WATCHDOG_VIOLATION,
a driver stalling) after about two hours of grids, losing a process 20 minutes in; the same bugcheck at the same code
offset was logged on 13 Oct, before the benchmark existed, so the load exposed the fault and did not create it. Claude
Code's memory-pressure reaper had also stopped the shell that chained the processes (the Python child survived). What
to do on a long run:
- one heavy process at a time, in chunks of about 24 configs, so that a crash costs minutes and not the grid
  (`run.py` writes nothing until its last config is scored);
- `OPENBLAS_NUM_THREADS=1` (and `OMP_`, `MKL_`): `play_features` multiplies matrices, which OpenBLAS spreads over every
  core; one thread gives byte-identical reports and the same `main.csv` (R013 checked both);
- below-normal priority for the venv's `python.exe` *and* its child, the real interpreter (the venv's is a launcher);
  a free-memory check before each process and a pause between them;
- check after a restart which logs end `EXIT=` and rerun only the processes that did not finish.

**T42 — A measurement run on a working tree that another agent is editing reads a half-written module.** The first
`restarts.py events` run of P17 failed with a `SyntaxError`: an implementer was in the middle of editing `restarts.py` in
the same working tree, and the run imported the file as it stood. Parallel agents share one working tree, so a run
started while one of them is editing reads whatever is on disk at that second (a run that does not crash may instead
measure code that was never committed). Launch measurement runs only at a committed, clean state of the files they
import, and record that commit in the entry; or run them from a separate worktree (`git worktree add`, with its own
`MTAH_DATA_DIR` if `dfl.py` differs). The failed run's output was discarded and the
command run again.

**T48 — A helper wrapper derived with `sed` silently kept the main venv's interpreter, and a lab's requirements file
changed the pipeline's libraries.**
P20's install of the separate `.venv-pathcrf` ran through a wrapper made from the main venv's helper by substitution.
One substitution did not match, the wrapper still pointed at the main `.venv`, and `pip install -r
requirements-pathcrf.txt` went into the pipeline's environment on 31 Oct from 01:33: numpy 2.4.6 became 2.2.6, pandas
3.0.6 became 2.2.3, scipy 1.17.1 became 1.15.3, matplotlib 3.11.2 was removed and torch was added. The venv was restored
from its pins and checked by about 01:40 (`selftest.py --data` passed at 01:40, `main.csv`'s sha256 was checked at
01:42 and unchanged, `7ba575c0…`). Commits landed inside the window and the suites run in it were voided and re-run;
no heavy run and no CSV came from that state. The output depends on those versions (T20, T21). Verify every
substitution by reading the derived file, never assume it, and make the first heavy command of a new venv print
`sys.prefix` and the interpreter's version before it installs anything.

**T49 — `git add -A <folder>` during a merge swept an untracked, gitignored-by-intent artifact into a commit.**
P19's trained model (3.6 MB, never meant to be committed) sat untracked in `pipeline/models/` when the merge of `main`
was committed with `git add -A docs pipeline`, and went into the merge commit. It was found before any push and the local
merge commit was amended, so no pushed commit holds it. Before committing after a merge or a training run, read
`git status --short` and add the files you mean by name; a model or a cache that must stay out of the history belongs in
`.gitignore` or in `data/` before it exists.

**T51 — A wrapper that starts Python with `Start-Process -NoNewWindow` can lose the exit code, and an empty code reads
as 0.** R027's gate printed "DROP (exit 1)", but the throttle wrapper's log ended `EXIT=` with no value, and the chain
that called it read the code as 0. The wrapper starts the venv's Python with `Start-Process -PassThru -NoNewWindow`,
waits with `WaitForExit()` and reads `$p.ExitCode`. Without the process's handle cached while it runs, that read can
come back empty: `"EXIT=$($p.ExitCode)"` then writes nothing after the `=`, and `exit $null` exits 0. It does not
happen every time. The same wrapper code logged `EXIT=1` correctly twice on 30 and 31 Oct, and P23's 27 runs all
logged `EXIT=0`, so a wrapper that has worked so far proves nothing. A verdict carried by the exit code was nearly
read wrong: R027's errors rule (a log without a valid exit is an error, not a verdict) caught it, and the step was
rerun once, 6 minutes. What to do:
- read `$p.Handle` (`$null = $p.Handle`) right after `Start-Process`, before anything waits on the process.
  Every `heavy*.ps1` in `.superpowers/bin/` (local, not in git) does so now: `heavy.ps1`, `heavy-p23.ps1`,
  `heavy-p22b.ps1`, `heavy-pathcrf.ps1` and `heavy-repro.ps1`;
- never let an empty code pass as 0: each of them logs `EXIT=UNKNOWN` and exits 99;
- read the tool's own printed verdict beside the log's `EXIT=` line, and treat a disagreement as an error.

**T52 — A cache keyed by a file's size and modification time serves a stale file rewritten to the same size within
one clock tick.** `trees.load_artifact` and `trees.remember` (`pipeline/evidence/trees.py`) reuse a parsed model while
its file's (size, `mtime_ns`) stamp is unchanged. `test_shape_possession_fold_choice` rewrote its shape model a few
milliseconds after writing it, with leaves of the same byte length, and Windows file times move with the system
timer's tick. When both writes fell in one tick the file looked unchanged, the stale model was served and the test
failed: 2 runs in 60. A probe of 300 quick same-size rewrites kept the same stamp 230 times. The fix, `afa721a`, is in
the test alone: it sets the rewritten file's mtime a second later (`os.utime`), as a real retrain, minutes apart,
would; 0 failures in 520 runs. A test or a tool that rewrites a cached file must change its size or move its mtime
itself (P05's sibling test is safe only because its new model has another length). Cost: a failure in a suite run that
had nothing to do with the change under test (P22(b)'s Task 1), and a replay to explain it.

## Rules and data handling

**T25 — Data must never leave our machines the wrong way.**
- Never give match data to any LLM, Claude included. The organisers' clarification of 3 Nov 2026 forbids "uploading
  the provided data to AI services" ([rules.md](rules.md)). Since then we count the public ground truth and the
  organisers' baseline CSVs as provided data too, since they are labels of this match's frames: an LLM gets their
  scores and counts, not their rows.
- Files of intermediate steps may be given to the organisers (3 Nov), but not committed: the repository will be public
  and the data licence still forbids derived data that would rebuild the data.
- Nothing from `data/` goes into git, and neither does derived frame-level data. Keep caches and labels under
  `data/`, which is ignored. A new folder outside `data/` is *not* ignored, so check `git status` before every commit.
- Don't move the data to cloud GPUs or notebooks without checking the T&C, and never with AI assistant features
  switched on.

**T26 — Nothing about this fixture from outside the bundle.** Event feeds, possession stats, match reports and goal
times are "human-produced data about the provided game, from any source" (RULES.md rule 2, as clarified on 23 Oct
2026, `f8f2629`, which names event data, corrected tracking, possession data and annotations). The same kinds of data
about other games are fine. Inferring the private GT's statistics from the published baseline scores is off-limits in
spirit too: tune on our own validation.

**T27 — No hand labels on this match, not even to validate, and the public window is not training data.** The
organisers' rule came in two clarifications, 22 Oct 2026 (upstream `c073f29`) and 23 Oct 2026 (upstream `f8f2629`),
both in RULES.md (rule 1, "Training and validation data", and rule 2). The emailed clarification of 3 Nov 2026
([rules.md](rules.md)) now takes precedence over both, and it narrows what is still fine (below, and
[T54](#rules-and-data-handling)). Hand-labelled data from other games is fine for training or validation, in any amount,
if we have the rights to use it, the organisers can access it and the technical document says what was labelled, how,
by whom and how much. Hand labels on BAR-ATH are not, in any amount and for any purpose, and nor is "any other data
created manually on the provided game".

*Changed 23 Oct 2026 (`f8f2629`):* until then the published 5 minutes could be used, and a model trained on them had to
predict the window out of fold, so that no ground-truth label was copied into the file. That reading is gone. The
5 minutes are for checking that the pipeline is correct: "They are not training data and not a tuning set." So there
is no out-of-fold use of them either; nothing is fitted, calibrated or chosen on them. The rule also grew a second
source of data: nothing human-produced about this game, from any source.

Not allowed on this match:
- labelling possession or stoppages from its video, including "blind" calibration on the public window;
- drawing ball boxes on its frames to fine-tune a detector (P14);
- hand-correcting any output ("frame by frame, segment by segment or at transitions only");
- correcting its tracking by hand (a ball position, an identity switch): "Corrections must come out of code";
- training a model on the public GT, or fitting, calibrating or choosing a constant on it, in sample or out of fold;
- any human-produced data about this game from any source: event data, corrected tracking, possession data,
  annotations (T26).

*Changed 3 Nov 2026 ([rules.md](rules.md)):* two things this trap listed as fine are not any more. Learning on this
match without human labels (unsupervised, self-supervised, pseudo-labels our code made) is out: "If you need to train
(or retrain), you MUST use only external data" (T54). Watching the video to debug the pipeline is out too: "checking
your output against the video" is not allowed, and watching is "to better understand what counts as possession".

Also not allowed on this match, since 3 Nov:
- training, retraining, fitting, calibrating or tuning anything on it, labels or not, and on its video by name (T54);
- comparing our labels, transitions or detectors' events with its video.

Still fine on this match:
- inspecting the data and watching the video "to better understand what counts as possession". What is not fine is
  checking our output against it, or turning what was seen into labels, corrections, or rules and constants tied to
  specific frames of the game, so the person of P02 watches the truth's changes to understand them, looks at none of
  our output, and writes no number into a config;
- running automatic checks using only the provided data: the file is valid, frames are aligned, the run is
  deterministic, a gate passes or fails, a score on the public 5 minutes (T01). A check never ranks or sets a value;
- using its video, tracking, Smart Tagging and other generated data as inputs, as long as nothing was trained on its
  ground truth or video (and, by T54, nothing was fitted on the rest of it);
- labelled data of other games we have the rights to use (P01, P13, P16).

Disclose every labelled dataset in the technical document, with its source, who labelled it and how much (section 2),
and none of it from BAR-ATH.

**T54 — Label-free is not enough: since 3 Nov nothing is fitted on our match at all.** Until 3 Nov RULES.md allowed
"learning without human labels" on the provided game, and our docs drew the line at labels and at the public window.
R017 was built on that line: it measured Smart Tagging's agreement with our own touch-rule pseudo-labels on BAR-ATH
(0.761) and wrote it into `main.toml` as `c_team` 0.76, with no label and no truth in sight. The organisers'
clarification of 3 Nov ([rules.md](rules.md)) moves the line: "If you need to train (or retrain), you MUST use only
external data. If your algorithm has parameters that need to be tuned, you MUST tune them on external data". So a
number measured on this match, from its tracking, tagging, homography, video or our own output on it, may no longer
set a constant, a weight, a prior or a model, however label-free it is.
- *The test:* does a number go from BAR-ATH into a config, a model file, a prior or a threshold? Then it is fitting,
  and it needs external data, a fixed rule (the laws of the game, the literature) or a fact the organisers document
  (the kick-offs). A fixed rule that reads BAR-ATH at run time (an episode, a team's mean position, a Viterbi path) is
  the algorithm running, which is fine.
- *Checks are still fine:* a label-free statistic of our output, a transfer gate, a score on the public window may pass
  or fail a candidate. They may not rank candidates or supply a value (T01).
- *What it caught on 3 Nov:* `smart_tagging` c_team in `main.toml`, taken out the same day (R029); P14's plan
  to train a detector on our video with the ATD's ball as pseudo-labels; P22(c). The benchmark's degradation, fitted on
  label-free statistics of our tracking (A07 to A09), was our reading's open case and question 1 for the organisers.
  Their reply of 5 Nov did not answer it and said "anything that is not explicitly allowed is not permitted", so by our
  reading it fails this test too: its constants go from BAR-ATH into the code that makes the training data (T56, P27).

**T55 — A `SET PIECES` tag is a fixed 35 s clip, not a window of a few seconds around the restart.** Measured on our
tagging on 4 Nov 2026 (`starter/load_smart_tagging.load`, lengths only): 57 of the 59 `SET PIECES` intervals (every
throw-in, corner, goal kick and free kick) are exactly 875 frames long, and the 6 kick-off tags exactly 125 frames
(5 s). T17 and CLAUDE.md said "a window of a few seconds around an event", which is true of the kick-offs only. So a
set-piece tag's start is a fixed offset before the restart (another team measured the ball out of bounds until a
median 6.3 s after a throw-in tag's start on this match, A22) and its end says nothing about play. That is why A11's
`tag_start` and `tag_end` detectors, which looked for a tag edge within 5 s of a stoppage's edges, found almost none,
and why R017's dead votes from the tags agreed with the dead-ball model on only 0.409 of their frames. Anything that
reads a set-piece tag as a short window is mis-specified; read it as "a restart happens a few seconds after this
instant", and set no constant for the offset on this match (T54).

**T56 — An open rules question is not a yes: the organisers answer with the clarification, and what is not allowed
explicitly is not permitted.** On 3 Nov we read the benchmark's degradation as allowed and asked the organisers to
confirm (question 1), and we kept building on it meanwhile. Metrica's unlicensed sample data had the same status
(question 3). Their reply of 5 Nov answered no question by name: it repeated the clarification and added "anything that
is not explicitly allowed is not permitted" ([rules.md](rules.md)). By then both models and every constant of
`main.toml` stood on the degradation, and R024's objective on R023, so the open case had become eight days of possible
rework before the deadline (P27).
- *The rule:* if no text allows something by name (the clarification, the challenge docs, the T&C), treat it as
  forbidden from the day it comes up, not from the day an answer comes. Asking is still worth it, but plan as if the
  answer will be the clarification again.
- *For a dataset:* "any dataset you have the rights to use. Please check the license". A repository with no licence
  grants no rights, whoever publishes it, and a README's request to acknowledge the source is not a licence.
- *For a measurement of our match:* a number that goes from BAR-ATH into anything that trains or chooses (a config, a
  model, a prior, a threshold, or the data a model is trained on) is fitting (T54), even when the code it goes into is
  not the algorithm.
