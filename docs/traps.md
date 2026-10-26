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
  frames (T27): a constant that fixes our convention would be tuned on the public window.

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
- Windows open seconds early (kick-off tags by ~3.5 s), and phases are not possessions.
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

**T43 — An oracle that stops where the truth changes measures more than a source can deliver.** A12 sized P17's
prize with a perfect 2 s vote for the restarting team (+0.0198 raw spellF1), and that vote stopped at the truth's next
change of team. A real vote has a fixed length and cannot know that change: it keeps pushing after the restarting team's
first possession has ended, and DFL's possessions are short (median 3.7 s). R014 delivered +0.0040 of the +0.0198 (and
raw spellF1 fell with every longer tail, to -0.0187 at 150 frames). Before taking an oracle's gap as a prize:
- build the oracle with the real source's footprint (a fixed span, no truncation at the truth's changes, no knowledge of
  where the truth's next possession ends), and read that gap as the ceiling;
- keep a perfect-information oracle as an upper bound only, and say in the protocol how far below it a source is
  expected to land.

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

## Rules and data handling

**T25 — Data must never leave our machines the wrong way.**
- Never give match data to any LLM, Claude included.
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
organisers' rule comes in two clarifications, 22 Oct 2026 (upstream `c073f29`) and 23 Oct 2026 (upstream `f8f2629`),
both in RULES.md (rule 1, "Training and validation data", and rule 2). Hand-labelled data from other games is fine for
training or validation, in any amount, if the organisers can access it and the technical document says what was
labelled, how, by whom and how much. Hand labels on BAR-ATH are not, in any amount and for any purpose.

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

Still fine on this match:
- watching the video and inspecting the data to understand them, debug the pipeline and design features. What is not
  fine is turning what was seen into labels, corrections, or rules and constants tied to specific frames of the game, so
  the person of P02 watches to understand and writes no number into a config;
- learning without human labels: unsupervised or self-supervised methods, and pseudo-labels our own code produces (P07),
  provided no person decided any of them;
- using the public 5 minutes to check correctness: the file is valid, frames are aligned, a source's agreement reads as
  it should, the run is deterministic (T01);
- labelled data of other games (P01, P13, P16).

Disclose every labelled dataset in the technical document, with its source, who labelled it and how much (section 2),
and none of it from BAR-ATH.
