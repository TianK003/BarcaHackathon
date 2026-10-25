# The P01 benchmark

We may not hand-label our own match, not even to validate (the organisers' clarification of 22 Oct 2026,
[CLAUDE.md](../../CLAUDE.md)), and the public window holds 17 possessions and ranks the baselines the wrong way round
([T01](../../docs/traps.md)). So the choices that matter (decoder, evidence weights, thresholds) are validated on
**other matches**: the seven open DFL/IDSSE matches (two Bundesliga, five 2. Bundesliga; CC BY 4.0), which carry the
ball's own alive/dead flag and the team in possession on every frame. A config runs on them through the pipeline's own
functions, is scored the way the organisers' scorer scores, and is compared with another config with the uncertainty
that seven matches allow.
The plan is [P01](../../docs/proposals.md); what was measured is in [results.md](../../docs/results.md) (A05 to A09,
R002 to R005).

```
python pipeline/benchmark/sources.py          # data -> data/external/dfl/   (21 files, 2.6 GB, size and md5 checked)
python pipeline/benchmark/run.py CONFIG ...   # configs -> scores, intervals, comparisons, one JSON report per config
python pipeline/benchmark/fidelity.py         # how far the degraded profile looks like our match (gates, warnings)
python pipeline/benchmark/headroom.py CONFIG  # a config beside five variants whose evidence is the truth itself
python pipeline/benchmark/selftest.py [--dfl] [--data]   # the checks
```

## Setup

```
# macOS/Linux: .venv/bin/python instead of .venv/Scripts/python
.venv/Scripts/python -m pip install -r requirements-benchmark.txt    # the pipeline's pins + lxml==6.1.3
.venv/Scripts/python pipeline/benchmark/sources.py                   # download, then check every size and md5
.venv/Scripts/python pipeline/benchmark/sources.py --verify          # only check what is on disk
.venv/Scripts/python pipeline/benchmark/sources.py --data-dir D      # another bundle (default data/, or $MTAH_DATA_DIR)
```

`sources.py` streams each file to `<name>.part`, hashes it as it arrives and renames it into place only if its size and
md5 are the pinned ones, so a file on disk is either complete and right or absent (a failed check is retried once from a
Hugging Face mirror of the same data). The files live in `data/external/dfl/<match id>/`, which git ignores; they are
public data and not the competition's, so unlike everything else under `data/` they may be read and printed.

**The data and its licence.** Seven full matches at 25 Hz, two Bundesliga and five 2. Bundesliga; five of the seven are
Fortuna Düsseldorf home games (J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9). Reference: Bassek, M., Rein, R., Weber, H.
and Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12,
195. https://doi.org/10.1038/s41597-025-04505-y. The data are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) "with the authorization from the DFL" and are on figshare
(DOI 10.6084/m9.figshare.28196177, as the paper's reference list gives it). We change the data (we degrade the
tracking) and say so. The data are public and outside the competition bundle, so their own licence governs them and not
the bundle's rules (the same reference is in section 2 of the
[technical document](../../docs/technical_document.md#2-assets-used-and-dropped)).

What to know about the files (A05; `dfl.py` has the details and refuses what it cannot be sure of):

- A match is three XML files: the match information, the raw events (never read) and the observed positions
  (350 to 420 MB each). `dfl.load_match(match_id)` is the one door to a match: it parses the positions once, stores the
  arrays under `data/cache/dfl/<match id>/` (129 to 141 MB per match, A05) and reads them back afterwards.
- **A first parse takes about 46 s** (21.6 to 64 s measured: the machine's load decides, A05); **a cached load takes
  0.7 to 1.9 s**, nearly all of it the sha256 of the two source files, which is what keeps a cache from being read
  after its sources change (A05). The cache is keyed on those hashes, on the numpy and lxml versions, on `CACHE_VERSION`
  and on the hash of `dfl.py` itself, so **any edit to `dfl.py`, a docstring included, rebuilds all seven caches
  (minutes)**: 174 s for the seven after a hash change, 326 s the first time (A05). The cache is one folder per match
  id and not one per key, so **one `dfl.py` per data dir**: two checkouts whose `dfl.py` differ that share a data dir
  rebuild each other's caches, and two rebuilds at once could mix their arrays, so give each its own `MTAH_DATA_DIR`.
  `load_match` also refuses a file of a registry match whose size is not the pinned one (`sources.py --verify` checks
  the md5).
- One timeline per match: the first half, 25 separator rows, then the second half. A row's place comes from the frame
  number `N`, never from the order of the file. Home and away come from each team's `Role`, never from the order of the
  teams in the file (the guest is listed first in four of the seven).
- `OtherGameInformation@TotalTimeFirstHalf` and `TotalTimeSecondHalf` are in **milliseconds**.
- Every squad member has a column in the position arrays, so **bench players are all-NaN columns** (7 to 11 of the 39
  or 40 per match: 8, 11, 7, 8, 9, 9 and 9, A05). `nearest_player` already ignores an absent slot.
- The ball is tracked on every frame of every half, and both flags are present on every frame, so the truth carries
  nothing and guesses nothing (A05, A06).

## Running a config

```
.venv/Scripts/python pipeline/benchmark/run.py \
    pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w25.toml \
    pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/nearest_w0.toml
.venv/Scripts/python pipeline/benchmark/run.py <the same configs> --truth coarse
```

| option | meaning |
|---|---|
| `CONFIG ...` | pipeline TOML configs (`pipeline/config.py`), per half (`scope = "halves"`). **The first one is the reference**: every other config is compared with it, and a delta is `other - first`, so positive means the other is better. Put the config you want to beat first. |
| `--profile P` | how the tracking is made (below): `tracking` (the default), `context` (a sensitivity check only: it fails two of its gates, R005, and the help, the printout and the report say so), `gaps` or `clean`. |
| `--seeds N` | degradation seeds per match (default 3: seeds 0, 1 and 2). The clean profile has nothing random to repeat, so it always runs seed 0 only; if you pass `--seeds` with more than 1 it says that it ignored it, and says nothing when you pass none. |
| `--matches ID ...` | the matches to run, in this order (default: all seven). Fewer than seven are filed under a name that says which (see the report, below). |
| `--truth raw\|coarse` | the truth to score against (below). Default `raw`. |
| `--data-dir D` | the data bundle (default `data/`, or `$MTAH_DATA_DIR`). |
| `--no-write` | print only. |

A run takes about 10 s on the clean profile with the caches warm (one cached load per match, a second or so per config);
the gaps profile runs each config once for each of its seeds. Everything that can be refused is refused before the first
match is loaded, with a sentence that names what to change: no config, a config that does not load, `scope = "file"`
(which decodes across half time; only `pipeline/configs/smooth50_repro.toml` needs it), a config whose file name has a
dot, two configs with the same name, an unknown profile, truth or match, a calibration that is not one.
A source that needs something a benchmark match does not have (the video's homography, the Smart Tagging) stops the run
with its config, its name and kind and the asset named; it is never scored as zeros.

### The yardsticks

`configs/nearest_w{0,25,50,125}.toml` are the organisers' four published baselines (`baseline_raw`, `baseline_smooth25`,
`baseline_smooth50`, `baseline_smooth125`: the nearest tracked player's team on every frame, then a centred majority
vote of +/- w frames) **decoded per graded half**, as every config of ours is. The published files are decoded over the
whole file at once, so these are yardsticks for the benchmark and not rebuilds of them (that is
`pipeline/configs/smooth50_repro.toml`). A change has to beat them here before it is worth anything.

P06's grid configs live in `configs/p06/`, named `p06s1_t{TTT}_d{DDD}.toml` for Stage 1: the hmm decoder with
`switch_team` TTT and `switch_dead` DDD (zero-padded to three digits), `nearest_player` at confidence 0.78 and
`ball_presence` at its defaults. The 24 files cross `switch_dead` d in {4, 8, 16, 32, 64, 128} with `switch_team` in
{d/4, d/2, d, 2d} (R006); later stages add their own prefix.

### The two truths

`--truth raw` is the DFL flags as they are: `D` where the ball is dead, otherwise the team in possession. DFL's flag is
fine-grained (437 possessions a match, a median of 3.7 s: A06), and how fine our ground truth is cannot be told from
the public window (A06, T33). So there is a second truth, `--truth coarse` (`truth.coarse_codes`): the raw truth with
every possession shorter than 5 s that sits between two possessions of the *other* team absorbed into it, a stand-in
for a coarser grain (1,659 possessions against 3,061, A06). **Run the yardstick under both and read the two side by
side**: a ranking that differs between them is a warning that the choice depends on the grain of the truth, not
something to fix. The two reports are filed apart (`<stem>.json` and `<stem>.coarse.json`).

### The profiles

A profile says how the tracking of a DFL match is made before the config sees it (`degrade.py`). A profile has a
version, which goes into every report: change what a profile does and bump it. Its constants are fitted once per
version, from label-free measurements of our tracking, and then frozen with the measurement in the code; they are never
tuned towards a gate, a WARN or a score (T26, T37).

| profile | version | the ball | the players |
|---|---|---|---|
| `clean` | 1 | DFL's, as it is: on every frame | DFL's, one slot per player |
| `gaps` | 1 | missing where **ours** was: our own missing-ball masks, transplanted onto each half with their pattern and not their clock (below); its speed is NaN everywhere (T31) | untouched |
| `tracking` | 3 | `gaps`'s, byte for byte | broken into tracklets of ours and given 46 slots, 24 home and 22 away, as ours are; a wrong-team slot now and then (10.5% of frames), NaN speeds in runs (7.8% of the slot-frames), 5-frame smoothing; **since version 2** only the players nearest the ball are tracked where ours has few (the cap), no piece is shorter than 8 frames, and the walk's spans are rescaled once (below). **since version 3** a slot is taken by a new tracklet only from the row after its last one stopped, and the scale is derived again (below); this is the default profile. Version 1 (R004) had none of the three of version 2, and versions 1 and 2 had the slot handovers that version 3 removes (R004c) |
| `context` | 1 | missing where our tracker loses it: a hazard fitted on our own match (A08) places the gaps by the ball's distance to a line, its speed, the players within 3 m of it (a crowd raises the hazard) and the minute of the half, with our gap lengths (`HAZARD_SCALE` 0.699, rescaled once). **It fails two of its gates**, the near-line share badly (its gaps start within 2 m of a line 56% of the time, against 28% of ours) and the missing share narrowly (R005), so it is **not the default**, and is kept for the agreement check between generators | `tracking`'s, draw for draw (`test_context_players_equal_tracking`) |

**The transplant.** `Calibration.ball_masks[half]` is our match's missing-ball mask of each half, from its kick-off to
the end of its graded block (M frames: 79,467 for the first half and 76,456 for the second, A07; `calibration.py`; the
Calibration is measured once and cached). A benchmark half of n frames takes the mask of our first or second half (1 or
2 with the same chance), read from an offset drawn uniformly over 0 to max(0, M - n) (the frames that leave room for
it; `degrade.choose_transplant`), so frame j of the benchmark half is missing where frame `offset + j` of ours was. A
half longer than the mask (n > M) has only the offset 0, and if the mask runs out, it goes on round the last 7,500
frames of ours: 3 of R003's 42 half-draws did, a DFL second half of 77,080 or 78,456 frames against our 76,456. In
short, the gaps are transplanted **with their pattern and not their clock**, and they are not placed by the play, which
is a separate matter:

- *What it keeps of ours:* the gaps' lengths and the pattern of the mask over a stretch of the half.
- *It does not keep the clock.* The offset is drawn uniformly, so the stretch is not at the benchmark half's own
  minutes (it is up to M - n frames later, 7.6 minutes for a 68,000-frame half against our 79,467-frame first half),
  and its share missing is the half's only on average over the offset, since coverage worsens through our halves (A02).
- *It does not keep where the gaps are in the play.* The mask is cut from our match and laid over DFL's play, so it is
  blind to it: a dead ball is no likelier to be missing than any other frame (in the public window ours is about three
  times as likely as a frame of the window in general: 33.6% of the dead frames against 11.8% of all of them, A01), and
  where on the pitch a gap starts is the DFL ball's own place at a random frame (36.0% of the gaps start within 2 m of
  a line, against 28.0% of ours, R003).

The warnings and statistics of the fidelity report show how much that matters.

**The tracklets (`tracking`).** DFL has one column per player. Ours has 46 anonymous slots for 22 players, and a slot is
a tracklet: it ends when the tracker loses the player and is taken over by whoever it finds next (T15). The profile
breaks each player's time on the pitch (each contiguous run of frames where DFL has a position) into tracked spans and
lost spans. A tracked span is one of **our own tracklet lengths** (`Calibration.tracklet_lengths_s`, at the match's
rate, at least 1 frame), drawn uniformly and multiplied by `TRACKED_SCALE` (below); a lost span is geometric (at least 1
frame) with the mean that makes the lost share d: d is 1 minus the mean number of slots present over the frames of ours
with more than 18 present, over 22, which is 0.0250 (A09), and the mean lost span follows the scaled tracked spans
(5.94 s against a mean tracked span of 231.8 s, J03WMX, R004c; before the scale 2.87 s and 111.9 s). Frames with 18 or
fewer present, 10.6% of ours, are the cap's (below) and are left out of d so as not to count them twice. All draws of
the walk come from `rng_for(seed, match_index, "tracklets", half)`, players by column and runs in time order.

**The wrong-team episodes (T32).** On 10.5% of our frames a team shows more than 11 slots while the other shows fewer: a
slot of one team carries a player of the other, because the tracker's identity followed the wrong one out of a duel.
Between the walk and the slot assignment (so the walk's draws are untouched), the profile cuts the tracked spans with
episodes of that kind. The onsets are a Bernoulli process over the rows of the half's graded block at the rate r = s *
target / mean episode frames per row, the target being `Calibration.wrong_team_share` and the episode lengths ours
(`Calibration.wrong_team_episodes_s`, both directions together, at least 1 frame). At an onset row, among the players
tracked then, it finds the closest home-away pair, in metres, ties to the lowest columns; **only if they are within 2.0
m** (identity swaps happen where players come close) does a random one of the two become an episode: his rows from the
onset to the end of the episode (or of his tracked span) form a separate tracklet in **the other team's pool**, and the
rest of his span is cut into a piece before it and a new tracklet after it (the split is an identity switch). An onset
with no such pair is skipped and counted (`skipped_far`), and so is one whose player is already in an episode
(`skipped_overlap`). Everything an onset needs is drawn up front from `rng_for(seed, match_index, "wrong_team", half)`,
in the order: one uniform per row (the onsets), one pick per onset, one duration per onset, so that no draw depends on
which onsets are skipped. Over the seven matches and three seeds of R004c (version 3), 5,330 onsets made 3,453 episodes
(64.8%) and skipped 1,746 as far (32.8%) and 131 as overlaps (2.5%) (R004, version 1: 3,445, 1,759 and 126). **s is
2.658, rescaled once** (it was 1.0): it calibrates the observable share, which `calibration.player_stats` measures as
ours was, and which is lower than the share of frames in an episode (a team shows a surplus only while the other team
has all 11 tracked and no episode of the other team runs the other way at the same time). At s = 1.0 the observable
share was 0.0397 against our 0.1054 +-0.03, so s became 1.0 * 0.1054 / 0.0397 and was frozen
(`degrade.WRONG_TEAM_SCALE`, with the measurement in its comment; R004). It is a label-free statistic and is never
touched again because of a WARN or a score.

**The speeds, and the smoothing.** The speed is a finite difference in m/s on the match's pitch, within a tracklet
piece, NaN on its first row. Then NaN-speed runs are placed: each starts at a present slot-frame chosen uniformly over
the half's present slot-frames, takes its length from our own runs (`Calibration.speed_nan_runs`, frames) and marks the
next L frames of that slot NaN, stopping at its tracklet's end; runs are placed one after another until the half's NaN
speeds, the first rows of the tracklets included, reach `Calibration.speed_nan_share` (7.83%) (`rng_for(...,
"speed_nan", half)`, the start and then the length for each run). The ball's speed stays NaN (T31). **Smoothing:** DFL's
players' jitter, the RMS of the second difference of a position (the statistic `calibration.player_stats` measured on
ours), is 8.996 cm in the seven matches, pooled by sums of squares, against 3.280 mm in ours: 27 times. The rule, set
before measuring, was to smooth every tracklet piece with a centred 5-frame moving average (the window shrinks at a
piece's ends to the rows inside it, and nothing crosses a piece) if DFL's is more than twice ours, so the profile is
smoothed (`degrade.SMOOTH_WINDOW`). On the profile's own draws (version 3, seed 0, the seven matches pooled; the same
pieces with and without the average, `test_data_tracking_constants_follow_from_the_measurements`) the average takes the
jitter from 8.75 to 1.77 cm, to a fifth of it (5.4 times ours, 3.28 mm), and the fidelity report's three seeds say 1.80
cm (5.5 times). Without the average the draws have 8.90 cm (three seeds), DFL's own 9.00 cm again; inside a piece the
average does what the arithmetic of a jump says, a step of J between two frames being two triples of +-J and, after the
average, two of +-J/5 (white noise falls to 16%). **The 6.47 and 8.5 cm that versions 1 and 2 reported were not DFL's
jumps: they were slot handovers** (below), a slot taken by another player on the very row it was freed, 13.8 m apart at
the median, 372 times in version 2's 21 runs. Measured inside the tracklets the same draws had 1.75 cm (version 2) and
1.80 cm (version 1), and the triples that straddle a handover were 96% and 92% of the squares per slot after the average
(R004c). Since version 3 no slot changes player between two adjacent rows, the jitter per slot is the jitter inside
tracklets (asserted), and no jump filter is needed. Five of the seven matches (all the 2. Bundesliga ones) have 98% of
the squares of their second differences in the largest 0.05% of triples: rare jumps (in J03WOY's first half, 61 steps of
1,520,419 exceed 25 m/s) that the average lowers to a fifth and does not remove. And the bulk is a 1 cm grid: the median
second difference of every match is 1.00 cm and its 99th percentile 2.83 cm, against 1.4 mm and 9.6 mm in ours, which a
5-frame average lowers by a factor of about 6 on white noise. Nothing was tuned to close the gap that is left (R004,
R004c).

**The cap, the shortest piece and the scale of the tracked spans (version 2).** Our tracker has 18 or fewer slots
present on 10.6% of its frames (5.2% have 14 or fewer): the camera is on a part of the pitch and the rest of the players
are not tracked. The cap (`degrade.cap_limits`, `cap_pieces`) puts that into the benchmark between the episodes and the
slots: on each frame where OUR count k is at most 18, only the k pieces nearest the ball are kept and the rest are
dropped on that frame. Our counts are read through **the same Transplant as the ball's mask** (same source half, same
offset, the same cyclic tail), so the cap and the ball's gaps come from the same frames of ours, which keeps what ties
them (both follow our camera); the ball is DFL's clean one, standing in for where our camera points, and where DFL has
none its last place in the half, then the centre spot. Membership is sticky: within a run of frames at or below 18 the
pieces kept stay kept, and it changes only when the count requires it (it falls: the farthest go; it rises or a kept
piece ends: the nearest present piece that is not kept comes in), never because two distances swap; at the run's end
everybody is back. A piece that loses frames is cut into the pieces between them, each a new tracklet that takes the
lowest free slot, so a player who comes back after a drop may not get his own slot back (the first thing the cap does to
slot identity, R004b). The cap draws nothing, so no stream of the other parts moves. Over the seven matches and three
seeds 10.1% of the frames are capped (ours 10.6%), 2,363,692 slot-frames are dropped and 4,513 pieces are cut.

**No piece is shorter than 8 frames** (`degrade.MIN_PIECE_FRAMES`, `drop_short_pieces`): ours has none (A09: the
shortest tracklet is 8 frames, 0.32 s), and version 2 without this rule made 2.2% of its pieces so (906 of 40,866;
195 of them were already short before the cap), mostly where our count flickers across the threshold (147 of the 155
crossings of 18 are a step of one slot) and the dropped players come back for a few frames and go again. After the cap
and before the slots every piece shorter than 8
frames is dropped, whatever made it (the walk's run ends, an episode, the cap), the player being absent there. Over the
21 runs it drops 823 pieces, holding 3,268 slot-frames, 169 of them episode pieces (`short_pieces`, `short_slot_frames`,
`short_episode_pieces` in the draws' record).

**The slots (version 3).** The pieces of a half, after the cap and the shortest-piece rule, are taken in order of (start
frame, player column) and each takes the lowest free slot of its team's pool. A slot is free for a piece that starts at
row t only when its last tracklet stopped before t: its half-open stop is at most t - 1, so there is at least one empty
row between two tracklets of a slot and a slot never changes player between two adjacent rows. Versions 1 and 2 freed a
slot at its tracklet's stop and let a piece that starts there take it, which put the position of another player into the
slot's trajectory: 372 times in version 2's 21 runs (7 matches x 3 seeds; 17.7 a run, 9 to 30), the two players 13.8 m
apart at the median and 40.1 m at the 90th percentile (version 1: 236 times, 5.3 and 43.0 m), where ours has none (3.28
mm of jitter over 3.2 million triples, A07). A review of R004b found it and R004c measures it. A pool with no free slot
is a ValueError (a team has 11 players on the pitch and 22 or 24 slots, so it means that DFL gave positions to players
who are not playing). `test_a_slot_never_changes_player_between_adjacent_present_rows` pins the rule on made-up matches
and `test_dfl_a_slot_never_changes_player_on_a_real_match` on a real one (J03WMX: no slot changed player on any of
11,526,850 adjacent present pairs of 4 seeds).

**The scale of the tracked spans** (`degrade.TRACKED_SCALE` = 2.072, frozen; 2.042 in version 2). The walk draws its
tracked spans from our *final* tracklets, which already hold the cuts that the cap and the episodes add, so the profile
counted our cuts twice: pooled over the seven matches, the walk alone gave a tracklet median of 40.88 s against ours
41.28 s, and each part that cuts took it further down (the episodes alone 29.26 s, the cap alone 24.44 s, both 19.68 s,
three seeds, R004b, with version 2's slot rule). The drawn lengths are therefore multiplied by lambda once (the lost
spans follow through mu, so the lost share stays d): lambda = ours' median over the profile's pooled median at lambda =
1 with the shortest-piece rule in (seed 0, the seven matches laid end to end as fidelity pools them). With version 2's
slot rule that median was 20.22 s over 13,192 tracklets, 41.28 / 20.22 = 2.0415, 2.042 (R004b); with version 3's it is
19.92 s over 13,396, because the old rule joined 204 pairs of tracklets into one run of a slot (a run of present rows is
one tracklet to `player_stats`), and 41.28 / 19.92 = 2.0723, **2.072**, set on 2026-10-23 (R004c). The bug that the slot
rule fixed had moved the measurement, so the constant was reopened once for that, and not for a gate, a WARN or a score;
it is applied whether or not the tracklet-median gate would pass, and it is never touched again for any of them
(`test_data_tracking_constants_follow_from_the_measurements` re-derives it from the recorded measurements and measures
that again). The pooled median of version 3 is then 26.52 s (three seeds), 36% below ours and inside the gate's 50%
(version 2: 27.08 s, 34%); the lost share of the walk, the episodes and the rule is 2.72% against d = 2.50% (seeds 0 to
4; version 2: 2.71%; the check allows 0.3 points: 2.59% before the scale and the rule, since a longer span makes the cut
at the end of a run weigh more).

The checks that pin the parts: `test_cap_keeps_nearest`, `test_cap_inactive_above_threshold`,
`test_cap_is_deterministic_and_draws_nothing`, `test_cap_membership_changes_only_when_the_count_requires_it`,
`test_cap_ball_fallbacks`, `test_cap_matches_the_documented_rule_on_random_pieces` (against a plain-loop reference),
`test_cap_leaves_min_of_the_count_and_the_tracked_players_on_every_row`, `test_cap_drop_breaks_slot_identity`,
`test_cap_limits_are_the_transplants_window_of_our_counts`, `test_cap_reads_dfls_ball_and_not_the_degraded_one`,
`test_cap_limits_refuse_counts_that_are_not_the_masks_frames`,
`test_transplant_counts_read_the_same_window_as_the_mask`, `test_min_piece_rule_drops_every_piece_shorter_than_it`,
`test_min_piece_rule_runs_after_the_cap_and_before_the_slots`, `test_min_piece_rule_counts_the_episode_pieces`,
`test_tracked_scale_multiplies_the_drawn_lengths_and_the_lost_share_stays_d`,
`test_tracked_scale_is_the_frozen_one_and_follows_from_the_recorded_measurements` (2.072 from 41.28 s and 19.92 s, and
version 2's 2.042 from 20.22 s, in the default run),
`test_calibrated_parameters_read_the_scale_and_the_shortest_piece_when_called` (the constants are read when the function
is called) and `test_tracking_with_no_low_count_frame_is_the_players_without_the_cap` (with no frame at or below 18 the
cap changes nothing, for the profile's parameters and for version 1's, lambda 1 and no shortest-piece rule, under
version 3's slot rule). The order of the parts is pinned by two scenes:
`test_cap_is_applied_to_the_pieces_after_the_episodes_and_leaves_their_counts_alone`
(an onset on a capped row is still counted, as the cap comes after the episodes) and
`test_cap_cuts_the_smoothing_and_the_speeds_at_the_drops` (nothing is smoothed or differenced across a drop). The rest:
`test_wrong_team_starts_at_close_pairs`, `test_wrong_team_moves_one_player_to_other_pool`,
`test_wrong_team_draws_come_from_the_wrong_team_stream_in_a_fixed_order`, `test_speed_nan_share`,
`test_speed_nan_runs_follow_the_documented_draws`, `test_smoothing_is_a_centred_moving_average_inside_each_piece`,
`test_profiles_nest`, `test_tracking_without_the_new_parts_is_the_walk_in_the_lowest_free_slots` (with the new parts off
the players are the walk's, each tracklet in the lowest free slot of its pool, byte for byte against a reference written
out), and on the downloaded matches `test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches`
and `test_data_tracking_constants_follow_from_the_measurements` (the rules that gave the scale of the episodes, the
smoothing window and the scale of the tracked spans, applied again). The earlier ones stay:
`test_tracklets_never_overlap_in_a_slot`, `test_each_tracklet_takes_the_lowest_free_slot_of_its_pool`,
`test_slot_team_matches_pool`, `test_tracking_degenerate_equals_gaps` (one endless tracklet per stretch of a player is
`gaps`'s players), `test_lost_share_on_long_synthetic`. Version 3's slot rule is pinned by
`test_a_slot_never_changes_player_between_adjacent_present_rows`,
`test_a_handover_on_the_row_a_slot_is_freed_is_what_the_check_would_catch` (the check of the check) and, on a real
match, `test_dfl_a_slot_never_changes_player_on_a_real_match`;
`test_each_tracklet_takes_the_lowest_free_slot_of_its_pool` and
`test_tracking_speeds_are_finite_differences_within_a_tracklet` now require that no slot is taken over on the row it was
freed, and `test_jitter_inside_tracklets_counts_what_it_says` pins the figure that the data check prints beside the
per-slot jitter.

**The `context` profile** (version 1; `degrade.context_probabilities`, `_context_ball`, `HAZARD_SCALE`). The `gaps`
profile transplants our masks with their pattern and not their clock, and is blind to the play. `context` keeps
`tracking`'s players, draw for draw, and places each half's gaps with the hazard that the Calibration fitted on our own
gaps (A08): a logistic model of six features on the 91,840 at-risk frames of our kick-off spans and their 143 onsets,
with cross-half AUCs of 0.71 and 0.65. Per half, from its kick-off to the end of its graded block, `hazard.features` is
computed in metres on the clean DFL ball and the degraded players: the distance to the nearest line, whether the ball is
beyond it, its speed over the last five steps, the nearest player, the players within 3 m, and the minute of the half.
The chance that the next frame starts a gap is `hazard_c` x `HAZARD_SCALE` x the model's probability, at most 1 - 1e-9,
and 0 where a feature is undefined (42 rows of 3,007,932 over the three seeds, R005). `hazard.sample_mask` draws the
gaps from `rng_for(seed, match_index, "hazard", half)`: the class of an onset is "within 2 m of a line at the frame
before it", its third of the half is that frame's, and its length is one of ours for that class and third at the match's
rate. The ball is then made by the function that makes `gaps`'s (x and y together, the speed NaN). **`HAZARD_SCALE` is
0.699, rescaled once.** At 1.0 the share missing was 0.4997 against our 0.4110, and the rescale is on the odds, 0.6986
(gaps start only from visible frames, so the odds are about linear in the hazard and a plain ratio would undershoot); it
is frozen, with both measurements, in the comment of the constant
(`test_hazard_scale_is_the_frozen_one_and_follows_from_the_recorded_measurements`).

**`context` fails two gates, and is not the default (R005).** Over the three seeds the share of the gaps that start
within 2 m of a line is 0.5586 against our 0.2797 +-0.05 (0.5647, 0.5644 and 0.5465 on the seeds: a fail on each), and
the share missing is 0.4313 against 0.4110 +-0.02 (0.4236, 0.4275 and 0.4427: a fail pooled, by 0.0003, and on seed 2).
Seed 0, where the scale was set, is itself 0.0126 above ours after the odds rescale, so most of the pooled +0.0203 is
the residual of that approximation and the rest the spread across seeds (0.0165 and 0.0317 above on seeds 1 and 2),
which were not used to set it. The cause is not the model, whose hazard is 2.9 times higher near a line on ours and 3.9
times on DFL's rows (the same ratios at any scale), but the rows that it is applied to: **DFL tracks the ball out of
play**, so 25.2% of the benchmark's at-risk rows are within 2 m of a line against 9.0% of ours, and 17.7% are beyond one
against 4.1%. Two remedies were measured and not applied: a chance of 0 beyond a line gives a near-line share of 0.163,
and a model that sees the distance clipped at 0 gives 0.489. Thinning the chance beyond a line to our share of such rows
(a factor of 0.23) was rejected: it would pass the gate by leaving even more of the out-of-play ball visible, which
moves the visible rows further from ours, and so would be tuning towards a gate; a mask for the ball out of the pitch
would be a new mechanism with a parameter that cannot be measured on ours (where the ball is during our gaps is
unobservable). There was no second rescale of the scale either: the rule is once. The profile is kept for the check that
the ranking of the configs does not depend on the generator: `context` and `tracking` version 3 give the same order
under both truths (Kendall tau +1.000 between the two orders), with every config 0.3 to 2.2 points of spellF1 higher
on `context` (R005).

**No truth reaches the degrader.** It is handed a `TrackingView`, which has the tracking and the match's scalars and no
`ball_status` or `ball_possession` (`test_degrade_never_sees_truth`, which also reads the degrader's source, and the
deep walk of the inputs, `test_degrade_serves_inputs_without_the_truth`). Its random draws are
`rng_for(seed, match_index, component, half)`, one stream for each, with no profile and no loop order in the key
(`test_degrade_independent_of_match_order`), so a match gets the same degradation whichever matches are run with it, and
the `tracking` profile, which degrades the players as well, has exactly `gaps`'s ball.

**A new calibration, a new report.** The report holds the digest of the Calibration its gaps were taken from. A report
is named by what was run (below), not by which calibration, so a later run with a newer Calibration writes over the
older report of the same name; compare the digests before comparing two reports.

### Fidelity: does the benchmark look like our match?

```
# -> data/runs/benchmark/fidelity_tracking.json
.venv/Scripts/python pipeline/benchmark/fidelity.py --profile tracking --seeds 3
```

`fidelity.py` degrades each match under seeds 0 .. N-1, measures it with **the same two functions that measured ours**
(`calibration.ball_stats` and `player_stats`), pools over matches and seeds (counts add, arrays are laid end to end: a
median is the median of every gap), and sets the result against the Calibration. Two kinds of criteria, fixed before
anything was measured and never tuned towards:

- **Gates**: the command exits 1 if one fails, and a profile can be the default only if all its gates pass (`context`
  fails two and is kept as a sensitivity check, R005). Each is our Calibration's
  own value, measured on the kick-off spans (a benchmark half starts at its kick-off, so both sides are measured alike),
  plus or minus the plan's tolerance. For `gaps`: the share of frames without a ball, +-0.02, and the median gap, +-1.5
  s. For `tracking`, those two and: the share of frames with exactly 22 slots present +-0.06, the mean number present
  +-0.5, the share with 14 or fewer present +-0.02, the wrong-team share +-0.03, the tracklet median **+-50% of the
  centre** (a relative tolerance; the rest are absolute) and the NaN-speed share +-0.01. For `context`, those eight and
  the share of the gaps that start within 2 m of a line, +-0.05 of the Calibration's own value (near-line onsets over
  interior gaps, pooled by sums). **The three gates on how many
  slots are present are measured on full-strength frames only**: those where the clean DFL match has exactly 22 players
  with positions, because a team that is down to 10 cannot show 22 slots whatever the degradation does (two of the seven
  matches have one, from minute 7 of the first half of J03WN1 and minute 13 of the second half of J03WQQ, and 18.3% of
  the pooled frames have 21 players: 81.7% are at full strength), and ours is measured on all its frames, which assumes
  that it has 22 players on the pitch throughout. Every other gate and every warning stays on all frames; the report
  holds the count statistics on both sets of frames, how many frames are at full strength and which matches are not all
  of them (`full_strength`). The plan's own figures (0.407, 10.7 s, 0.108, 35.5 s, ...) were measured on other spans and
  are not used. **Pending gates** (`fidelity.PENDING_GATES`) are printed and not counted, and none is pending now:
  version 1 of `tracking` had the three about how many slots are present (exactly 22, the mean, 14 or fewer) pending,
  because it had no low-count tail, and version 2 has the cap ("The cap, the shortest piece...", above), so they are
  gates like the rest. The mechanism stays for the next part of a profile that a gate waits for.
- **Warnings**: the nearest player's agreement with the truth on alive, ball-visible frames (our public window's,
  +-0.05), the share of dead frames with the ball visible (+-0.10), and for each yardstick its possessions per graded
  minute (+-25%) and its D share (+-0.05), against what the yardsticks output on ours (over our graded halves, as the
  published baselines are: an accepted offset of at most 1% of the frames). A WARN is printed and discussed in
  `results.md`; the degradation is not adjusted towards it, because some misses are the truth's conventions and not the
  degradation's (A06).

The report also lists the benchmark's other statistics next to ours (tracklets, wrong-team frames, NaN speeds,
jitter...), which `gaps` does not model, so that what a later profile has to fix is on the page, and the degrader's
draws (two integers a half). For `tracking` and `context` it holds the pooled counts of the players' draws
(`players_draws`, `fidelity.PLAYERS_DRAWS`: tracklets, onsets, episodes, skipped_far, skipped_overlap, episode_frames,
speed_nan_runs, speed_nan_frames, the cap's capped_frames, cap_slot_frames and cap_pieces_cut, and the shortest-piece
rule's short_pieces, short_slot_frames and short_episode_pieces), and for `context` those of the ball's
(`ball_draws`, `fidelity.BALL_DRAWS`: rows, ball_missing_rows, nan_feature_rows, capped_rows, onsets, missing_frames,
at_risk_rows and post_gap_rows, with the scale and two shares worked out from the sums, post_gap_share and
nan_feature_share). It holds no frame of our match, and it carries the Calibration's digest and the git state of the
code that made it (`git`: the commit and whether the tree was dirty, as the runner's reports have them). It is named
`fidelity_<profile>[.m-<ids>][.s<n>].json` by the runner's rule.

### What the clean profile cannot show

The DFL ball is tracked on every frame, so `nearest_player` votes everywhere and `majority_vote`, which takes `D` only
where nobody voted, labels almost nothing `D`, where the truth has 41% to 52% of every match dead. So on `clean`
the alive/dead F1 is the same for every yardstick (it is the F1 of "always alive", the scorer's ALIVE being the positive
class), the macroF1 is held down by an empty `D` class, and the ranking is only about who has the ball. That is
expected, and is why the degraded profiles (which give the ball our tracking's gaps) exist. It is not a bug of the
configs. `gaps` gives the ball our gaps, so that `majority_vote` has `D` to give, and the scores are no longer those of
a ball that is always there.

## Reading the printout

Per config:

- **pooled** spellF1, macroF1 and alive/dead F1: the official scorer's numbers for all the halves of all the matches as
  one file. They are computed from summed counts, never from an average of averages, and `score.py`'s checks assert that
  they equal `pm.evaluate` on the halves laid end to end.
- **per match**: the mean of the seven matches' own spellF1 and its 90% Student t interval. This one is not the
  scorer's number: it weighs each match once, and it is the one an interval can be put on.
- **possessions** per graded minute, predicted and true, and the share of graded frames labelled `D`, predicted and
  true. A config that flickers shows up here before it shows up in a score.
- spellF1 by match, and the seconds the config took (printed, never written to a report).

With two or more configs, each against the first:

- **pooled delta** and **mean per-match delta**, with the 90% t interval of the match deltas. An interval is said to
  **exclude 0** only if it clears 0 by more than `score.FLIP_TOLERANCE` (1e-12, the margin under which two scores are
  equal), because a delta of the order of 1e-16 can otherwise make a gap out of float noise.
- **sign-flip p**: the exact two-sided randomisation test of the seven deltas, with all 2^7 sign patterns enumerated.
  The smallest p it can give with seven matches is 2/128 = 0.0156, and a config that wins every match gets exactly that.
- **wins and losses**: the matches whose delta is above and below 0 by more than `score.FLIP_TOLERANCE`.
- **by competition** and **per match**: the same delta for the two Bundesliga matches and the five 2. Bundesliga ones,
  and for each match. Five of the seven are Fortuna Düsseldorf home games, so seven matches are not seven independent
  draws of "a match"; a gap that holds in every match and in both competitions is worth more than one that does not.

The uncertainty is **never a bootstrap**: resampling seven matches, or the frames inside them, would pretend to a sample
the benchmark does not have. An interval says how far to trust a gap; it is not a licence to prefer a config on a gap it
does not clear, and the public window is no tie-breaker either (T01).

### The report

`<data dir>/runs/benchmark/<profile>/<stem>[.<truth>][.m-<ids>][.s<n>].json`, one per config, gitignored with the rest
of `data/`. The name says what the report is of, so that one never replaces another of a different kind:

| run | file |
|---|---|
| all seven matches, raw truth, the default 3 seeds | `<stem>.json` |
| all seven, coarse truth | `<stem>.coarse.json` |
| any other set of matches | `.m-<the ids, sorted, joined by "-">` before `.json`, e.g. `<stem>.m-J03WMX.json`, `<stem>.coarse.m-J03WMX-J03WN1.json`. The order they were asked for in is not in the name; all seven in another order are still `<stem>.json` |
| a number of seeds other than 3 (a degraded profile) | `.s<n>` last, e.g. `<stem>.s1.json`, `<stem>.coarse.m-J03WMX.s5.json`: a quick one-seed look never replaces the three-seed report. The clean profile has one seed, always, and its name never says so |

A config's own file name may hold no dot: `x.coarse.toml` would write `x.coarse.json`, which is `x`'s report under
another truth, so the runner refuses it and says to rename it.

It is written with sorted keys, indented, LF, floats rounded to 9 decimals and no timing anywhere, to a temporary file
renamed over the old report; the same inputs, seed and versions give the same bytes in any process.

| key | holds |
|---|---|
| `config` | `path`, and `text`: the TOML as run, with LF line endings |
| `profile`, `seeds`, `truth` | the profile's `name` and `version` (and for `context` `tracking_version`, the version of tracking's players that it is built from), the seeds run, and `{name, min_s}` of the truth |
| `sensitivity_only` | only in the report of a profile that is a sensitivity check and not the yardstick (`context`): why, as `run.SENSITIVITY_ONLY` says it; absent for the others |
| `calibration` | `{sha256, version}` of the Calibration the tracking was degraded with (`calibration.digest`); null for `clean`, which degrades nothing |
| `matches` | per match, in match-id order whatever order they were run in: `id`, `frames`, `graded_frames`, `fps`, `carried_frames` (frames the truth had to carry: 0), and `sources`: the size and sha256 of the two DFL files it was made from (null if the match was not read through the cache, with `sources_reason`) |
| `git`, `environment` | the commit and whether the tree was dirty; the Python, numpy and lxml versions and the platform |
| `pooled`, `per_match` | the scores of all halves, and of each match: `spell_f1_mean`, `spell_f1` at each IoU, `macro_f1`, `alive_dead_f1`, `n_gt`, `n_pred`, `halves` |
| `t_interval` | the t interval of the per-match `spell_f1_mean` and `macro_f1`: `n` (the matches), `level` (0.9), `mean`, `low`, `high`. Only `low` and `high` are null, for fewer than two matches; the `mean` of one match is its own score |
| `possessions_per_graded_minute`, `labels` | the rates and the label shares of the printout |
| `halves` | the counts everything above was made from, per half: `n_gt`, `n_pred`, `tp` at each IoU, the 3x4 `confusion` of the unfilled codes |
| `versus` | null for the reference; for the others, the comparison with it for `spell_f1_mean` and `macro_f1` (`score.compare`: deltas, interval, p, wins, losses, by competition, and whether the interval excludes 0) and the reference's name and config hash. It is the one field that depends on which other configs were run |

## Headroom

What would a perfect dead-ball detector, or a perfect possessor, be worth to a decoder that is already there? Before
building P05 or P04 it is worth knowing how much of the score each could ever deliver. `headroom.py` runs a config on
the benchmark beside five variants of it whose evidence is the truth itself:

```
.venv/Scripts/python pipeline/benchmark/headroom.py CONFIG [--profile P] [--seeds N] [--truth raw|coarse] [--matches ID ...] [--data-dir D] [--write]
```

`CONFIG` is the reference, run from its own file as it is, and needs exactly one `nearest_player` source (the home
variants replace it). The runner's table then says, against it, what each variant gains:

| config | evidence |
|---|---|
| `<stem>__alive` | CONFIG's own, plus `oracle_alive`: a perfect dead-ball detector |
| `<stem>__home_visible` | `nearest_player` replaced by `oracle_home_visible`: the true possessor on the frames where the tracking has the ball, nothing where it does not |
| `<stem>__home_all` | `nearest_player` replaced by `oracle_home_all`: the true possessor on every live frame, the ball seen or not |
| `<stem>__alive_home_visible` | both of the above |
| `<stem>__alive_home_all` | both, whole: the truth itself, which a decoder free to follow it reproduces (exactly 1.0 on all three scores for an hmm with switch penalties near 0 and minimums of 1) |

The oracles are exact: `oracle_alive` is 1 on the truth's `H` and `A` and 0 on its `D`; the home oracles are 1 on `H`
and 0 on `A`, the visible one only where `atd.ball_present` (as the profile degraded it, so the ball is missing where
ours was), and every channel they do not speak on is NaN. A source that replaces `nearest_player` keeps its weight,
takes the name of its own kind and none of its parameters; `oracle_alive` is added after the config's own sources, at
weight 1. To the hmm decoder a perfect source is about 13.8 nats a frame at the default `eps`, so the table is a bound
on what the evidence could give this config and its penalties, and not an estimate of what a real source will.

- **The truth reaches the oracles by closure, never through the inputs.** It is `truth.truth_codes` of the match that
  the runner itself loaded (`truth.coarse_codes` of it under `--truth coarse`), found by `run.inputs.game`, so the
  oracle and the score agree on which truth it is. Without `matches` the runner is handed a `Mapping` that loads each
  DFL match when asked, one at a time, and keeps none, so a headroom run needs what any other needs.
- **Oracles exist only while a run lasts.** `registered` puts them into `pipeline.evidence.REGISTRY` for the length of
  the call and restores the registry afterwards, a run that raises included. Importing `headroom.py` registers nothing,
  and a config that names an oracle does not load outside it. `pipeline/decode/` and `pipeline/evidence/` know nothing
  of it.
- **The variants are files.** Written as TOML into `<data dir>/runs/benchmark/headroom/configs/<stem>__<variant>.toml`,
  a fixed folder, so a config's path and text (both are in its report) and its report's bytes are the same on every
  run. A file that already holds exactly the bytes is not written again, not even touched: the texts are the same under
  both truths, so the raw and the coarse run can go side by side, which they could not if each replaced files that the
  other has open (Windows refuses that). A parameter that is not a str, int, float or bool (the flat tables that
  `Config` holds) is refused by name before any file is written, and so is a config without one `nearest_player`. What
  only the runner knows (an unknown profile, say) is refused by it, after the variants are written.
- **Reports** are written only with `--write`, and only for the five variants, as the runner writes them
  (`<stem>__<variant>[.<truth>][.m-<ids>][.s<n>].json` under `<data dir>/runs/benchmark/<profile>/`). `CONFIG`'s own
  report is `run.py`'s to write: it holds the comparison with whichever configs it was run beside, which a headroom run
  cannot reproduce, so a headroom run never writes or replaces it.

Read it as an upper bound and a loose one: no real detector is perfect, and these two do not need the ball, which a
real one would. It says where not to spend time (a small gap is a small prize) and in which order to think about the
work. Run it under both truths and read the two together, as for any comparison, and across the seven matches rather
than in the pooled number alone. Nothing here is a config to submit. The checks that pin it are
`test_headroom_variants_and_order`, `test_headroom_full_oracle_is_perfect`,
`test_headroom_coarse_truth_reaches_the_oracles`, `test_headroom_oracles_say_what_the_truth_says`,
`test_headroom_visible_oracle_follows_the_degraded_ball`, `test_headroom_restores_the_registry`,
`test_headroom_never_in_production`, `test_headroom_variants_round_trip_what_config_holds`,
`test_headroom_toml_writer_reads_back_as_written`, `test_headroom_refuses_what_it_cannot_make_variants_of`,
`test_headroom_a_second_write_leaves_the_variants_alone`,
`test_headroom_a_write_that_loses_a_race_for_the_same_bytes_is_not_an_error`,
`test_headroom_loads_each_match_once_and_keeps_none`,
`test_headroom_writes_the_variants_reports_only_when_asked_and_never_the_references` and
`test_headroom_command_line_prints_the_table_and_the_report_lines`, all in `checks/headroom.py` and all synthetic.

## The checks

```
.venv/Scripts/python pipeline/benchmark/selftest.py          # synthetic: a minute or two
.venv/Scripts/python pipeline/benchmark/selftest.py --dfl    # + the seven downloaded matches (minutes)
.venv/Scripts/python pipeline/benchmark/selftest.py --data   # + our own match (aggregates only; minutes)
```

The synthetic run is 288 checks (P06's added 16 to the 272 of A05). Its time, 49 to 115 s, is as of A05 and not
measured again. `--dfl` and `--data` are as of A05 (2026-10-23, `fdc5eae`), when the synthetic run was 272 checks:
`--dfl` added 16 and `--data` 7, and all of them (295 checks then) took 949 s (15 min 49 s) with the seven DFL caches
and the Calibration rebuilt after an edit to `dfl.py` and `calibration.py`. The machine's load decides, and the first
run on a machine also builds the caches (A05).

The checks live in `pipeline/benchmark/checks/`, one module per topic, listed in `checks.MODULES` (the folder's
`__init__.py` says what each holds): `sources`, `dfl`, `truth`, `inputs`, `score`, `run`, `hazard`, `calibration`,
`degrade`, `cap`, `tracking`, `context`, `fidelity`, `headroom` (the Headroom section's) and `harness` (the harness
itself). The profiles' are `degrade` (the
transplant worked out exactly, determinism, what the `gaps` profile touches and what it does not, no truth, the order of
the matches), `tracking` (the tracklets, the episodes, the NaN-speed runs and the smoothing worked out exactly, and how
it nests in `gaps`), `cap` (the low-count tail's cap, its shortest-piece rule and the scale of the tracked spans),
`context` (what the hazard is given, the chance, the class, third and durations read at the at-risk frame, the counts of
its record on hand-built halves, the odds rescale and the frozen scale) and `fidelity` (every gate and check worked out
by other code, pooling by sums, the exit code, the pending gates, the full-strength frames of the count gates and the
gate kinds). The one that anchors the rest is `test_dfl_oracle_scores_perfectly`: an evidence source that is the truth
itself, decoded by `majority_vote` with half-width 0, must score exactly 1.0 on spellF1, macroF1 and alive/dead F1 on
every one of the seven matches.

**What the oracle checks, and what it cannot.** It checks the chain from the *loaded match* to the score: the runner,
`MatchInputs` and the separator rows between the halves, both halves and each one's first and last row, the decoder path
and the scoring. A row too many or too few at the half time, a lost first row, a half decoded one row late or a team
swapped on the way would not score 1.0. It does **not** check the link from the DFL file to the truth: the oracle's
answers and the runner's truth are the same `truth.truth_codes` of the same loaded match, and the oracle never reads
the tracking, so a mistake in the parse or in `truth_codes` is on both sides and cancels (with H and A swapped inside
`truth_codes` it still scores 1.0 on all seven matches). That link is covered by A05's checks of the parse against
kloppy and by frame number (`test_dfl_parser_agrees_with_kloppy`, `test_dfl_second_half_spot_check`), and for the teams
by `MIN_AGREEMENT` in `test_dfl_truth_conventions`: the nearest player must name the truth's team on more than 60% of
the alive, ball-visible frames of every match and of each half (it does, at 71% to 82%; with the teams swapped it would
be at 18% to 29%).

All three scores are asserted, exactly, because each is blind to something another sees: macroF1 counts frames, so it
sees a single wrong frame anywhere; spellF1 matches whole possessions by IoU, so it forgives a boundary a frame out
unless the possession is very short; alive/dead F1 sees only dead against alive, so it cannot see a team. A review's
mutant that left the first row of the second half undecoded (X) was seen by macroF1 and not by spellF1: the scorer
back-fills an X at a half's start for the possession metrics. That is not the slip of the controls, which shift every
label of a half by one row. The oracle's controls on J03WMX show that the check can see: one row late (both
halves, the second half only, the first half only) is below 1.0 on all three scores, a swapped team in the second half
leaves alive/dead F1 at exactly 1.0 and takes the other two far below it, and the raw oracle against the coarse truth
is below 1.0 (the coarse truth is applied).

The others pin the report's shape, that every number in it is what the scoring functions say, the direction of a
comparison, that a match is gone before the next is loaded, that the same run writes the same bytes in another process
with another `PYTHONHASHSEED`, that a subset of the matches never replaces a full report, what the runner refuses, and
that this README names only checks that exist.

## Limitations

What the benchmark cannot show, and where its copy of our match is not ours. Section 6 of the
[technical document](../../docs/technical_document.md#6-limitations), which the organisers read, lists the same limits,
and this is the detail behind that list. None of them makes a benchmark number ours: absolute spellF1 here is not our
leaderboard's, and a ranking that holds under both truths and in both competitions is worth more than one that does not.

- **Conventions and grain.** DFL's flags are one provider's, and ours may flip at other moments and cut possessions at
  another grain. DFL's flag is fine-grained, 437 possessions a match with a median of 3.7 s, and its `D` is its own
  ball-in-play clock to within 2 frames (A06). How fine our ground truth is cannot be told from the public window: the
  window is alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows, and against DFL's windows that lively
  it is 1.34 times slower, which DFL's own conventions can produce (A06, T33). `--truth coarse` stands in for a coarser
  grain and does not correct a measured difference, so a choice that depends on possession length (a smoothing level, a
  switch penalty) is ranked under both truths and a disagreement is a warning. `D` timing against ours is unmeasured,
  and the timing parameters come from the public window (P02).
- **Seven matches, five of them Fortuna Düsseldorf's home games** (J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9). The
  sample is small, so the intervals are wide, and it is not independent, so even they may be too narrow; with seven
  matches the smallest sign-flip p is 2/128 = 0.0156. A degradation seed moves a pooled spellF1 by up to about 0.015
  (R003) and 0.013 (R004c), which the intervals, being over the matches, do not show; that is why three seeds are the
  default.
- **A team of 10 in two matches.** J03WN1's away team has 10 players from minute 7.1 of the first half, and J03WQQ's
  from minute 13.4 of the second (T36); the open data gives no cause. A statistic on how many slots are present is
  measured on full-strength frames only (81.7% of the pooled frames), and the wrong-team episodes are about half
  unobservable on 10 against 11, which lowers that share in those two matches (R004).
- **No video, audio, homography or Smart Tagging.** DFL has none, so whatever uses the camera, the audio, the homography
  or the tagging is checked on the public window and not here, or simulated from the benchmark's truth with noise
  measured on our match (A03).
- **Aerial projection.** A camera-based tracker projects an airborne ball onto the pitch plane, which displaces it.
  DFL's positions carry no such error and the benchmark adds none, and our ATD has no ball height, so the error cannot
  be measured on ours. Nor can we tell how many of our ball gaps come from airborne balls: the plan said "crowded and
  aerial", and what was measured is "lines and slow balls" (A08).
- **No positional bias.** The profile brings the players' jitter down towards ours (next item) and does nothing else
  about their positional error. A slowly varying bias against the true position cannot be measured on ours (there is no
  reference to measure it against), so none is added. Published accuracies for video-derived positions run from 0.08 to
  0.09 m RMSE for a multi-camera stadium system against a motion-capture reference (Linke, Link and Lames 2020, PLoS ONE
  15(3): e0230179) to a median of 1.1 to 1.7 m (mean 1.8 to 2.8 m) for a pipeline on broadcast video (Theiner et al.,
  WACV 2022, arXiv:2110.11107, Table 3), so a bias of the order of 0.5 to 2 m is plausible for ours and would hurt a
  method that reads distances more than the benchmark says (the plan's range of 0.5 to 2 m is not stated by either
  source, which are the nearest published figures; it is not our measurement).
- **Rougher positions.** After the 5-frame smoothing the jitter is 1.80 cm per slot against our 3.3 mm, 5.5 times ours
  in RMS (T35, R004c). DFL's clean 9.0 cm is a few jumps on a 1 cm grid, and the average takes it down to a fifth; what
  is left is DFL's, lowered. A method that reads positions, speeds or accelerations sees rougher ones than ours, so
  compare medians and percentiles and not an RMS (T35).
- **The default's ball gaps are transplanted with their pattern and not their clock, and are blind to the play.** The
  offset is drawn uniformly, so the pattern lags by up to M - n frames (our half less the benchmark's), 7.6 minutes in
  a 68,000-frame half (R003's offsets reached 11,222 frames), and most of the climb through each half (A02) arrives
  shifted. Separately, the mask is cut from our match and laid over DFL's play: a dead ball is no likelier to be missing
  than any other frame, where in the public window ours is about three times as likely (33.6% of the dead frames
  against 11.8% of all of them, A01), and 36.0% of the gaps start within 2 m of a line against 28.0% of ours (R003):
  placed without regard to the play, they start near a line more often in a DFL match than ours do, and DFL tracks the
  ball out of play (R005). The default has no gate on that share, which is `context`'s, and 36.0% is outside the +-0.05
  that gate allows (R005). The ball is visible on 0.5684 of the benchmark's dead frames against 0.6637 in the public
  window (R003, R004c), which is far better covered than the match (12% of its frames without the ball against 41%), so
  ours over the whole match is unmeasured. A method that depends on where the gaps fall is validated on gaps that ignore
  the play.
- **`context` does not fix that, and its measured reason (R005).** It places the gaps by the play (lines, slow balls
  and a crowd of players round the ball, as ours are: A08) and fails two of its gates, the near-line one badly and on
  every seed: **56% of its gaps start within 2 m of a line against our 28%**. The cause is not the model, whose hazard
  is 2.9 times higher near a line on ours and 3.9 times on DFL's rows (the same ratios at any scale), but the rows it is
  applied to: **DFL tracks the ball out of play**, so 25.2% of the benchmark's at-risk rows are within 2 m of a line
  against 9.0% of ours, and 17.7% are beyond one against 4.1%. No remedy was applied (R005). The other gate it fails is
  the share missing, narrowly and pooled (0.4313 against 0.4110 +-0.02), and it warns on the ball visible on dead
  frames, 0.4622 against the public window's 0.6637 (`tracking` version 3: 0.5684): the hazard puts gaps on slow balls
  and a dead ball is slow (a reading, not measured), and the window is far better covered than the match, so some of
  that miss is the window's. The hazard is a model of six features fitted on 143 onsets of one match. Its ball features
  come from the clean ball, so on the five frames after a gap the speed is the true one, where ours carries the speed
  from before the gap: 0.83% of the at-risk rows are in that window (0.84%, 0.81% and 0.84% by seed), and the
  approximation is counted in the record (`post_gap_rows`). And `tracking` reads the cap's counts and the ball's mask
  from one window of ours, so both follow our camera, where here the cap reads a window of ours and the gaps are drawn
  from the play. So `context` is a sensitivity check and not the default.
- **Tracklets and slots are not ours.** The tracklet median is 26.52 s against our 41.28 s, 36% below ours and inside
  the gate's +-50% (R004c, T37). A slot's index is not as empty of the player as in ours. The profile gives each
  tracklet the lowest free slot, so a team uses fewer of its slots than ours does over a whole match (J03WMX, seed 0,
  version 3: of the pools of 24 and 22, 13 and 13 slots are ever present, 12 and 12 on at least a tenth of the frames
  and 13 and 12 on at least a hundredth, where ours has 22 and 21, 21 and 20, 22 and 21; printed by
  `test_data_tracking_profile_on_a_real_match_with_our_real_calibration`). A player who comes back after a lost span
  lands in the slot he had 47.4% of the time (529 of 1,117 returns; 48.1% of 1,147 in version 2, 69.0% in version 1,
  R004, and 67.6% of 562 returns with the walk, the episodes, the scale and the shortest piece but no cap), and a slot
  idles between two tracklets for a median of 4.3 s (39.4% of its gaps at most 3 s, 5.3% over 60 s), where ours idles
  for 7.4 s (17.7% and 15.0%; A09, R004c; version 2: 4.1 s, 42.0%, 5.3%; version 1: 2.3 s, 59.7%, 3.8%). No evidence
  source reads a slot's identity today, beyond `nearest_player`'s argmin on exact ties, which goes to the lower slot;
  but **a method that follows a slot across a loss would be validated optimistically here**, a little less than before
  the cap. The episodes are cut at pairs within 2 m only, in both directions at one rate, and what is matched is the
  observable share, not the episodes: on seed 0 the benchmark shows 75 to 159 observed episodes a match (mean 124.6),
  medians of 2.0 to 2.8 s and 90th percentiles of 8.2 to 14.5 s (R004c; version 2: 82 to 151, mean 121.6, 1.8 to 2.7 s
  and 9.4 to 14.4 s), against ours 104, 2.7 s and 15.8 s (a swap inside 11 against 11 is invisible on ours, so the true
  number of ours is unknown).
- **The cap keeps the players nearest the true ball.** Where ours has 18 or fewer slots present, the profile keeps the
  nearest players, so a method that decodes from the nearest player to the ball is hurt little by it (R004b: the four
  yardsticks move by 0.4 points of spellF1 at most), and one that needs the players far from the ball sees fewer of
  them than the real match gave it, as it does on ours. The cap takes our counts from the frames that the ball's gaps
  come from, not from the same play, so what ties the two to the play is not in them.
- **The halves.** A benchmark half starts at its kick-off, while our graded second half starts 117 s before it, all `D`
  (A06, T02). DFL's halves start alive in 11 of 14 (the other three have 1, 1 and 48 leading `D` frames) and all 14
  end in `D`, so the benchmark cannot test the first and last seconds of a half as ours are (A06).
- **`clean` and `gaps` are floors, not copies.** `clean` has the ball on every frame and DFL's players, one slot per
  player. `gaps` has our gaps and DFL's players, with no wrong-team slot and no NaN speed. The fidelity report's
  statistics say by how much each is short of ours, and "What the clean profile cannot show" says what that means.
- **Where the transplant is cut.** A transplanted mask is cut at the edge of the window it is read from, so up to two
  gaps of a benchmark half (the ones at its ends) are fragments of ours, and where the mask runs out and goes round
  there is one more at the seam (3 of the 42 halves of R003: a DFL second half of 77,080 or 78,456 frames, against the
  76,456 of ours); the gates are on the share and the median, which this barely moves.
