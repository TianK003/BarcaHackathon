# The P01 benchmark

We may not hand-label our own match, not even to validate (the organisers' clarification of 22 Oct 2026,
[CLAUDE.md](../../CLAUDE.md)), and the public window holds 17 possessions and ranks the baselines the wrong way round
([T01](../../docs/traps.md)). So the choices that matter (decoder, evidence weights, thresholds) are validated on **other
matches**: the seven open DFL/IDSSE Bundesliga matches (CC BY 4.0), which carry the ball's own alive/dead flag and the
team in possession on every frame. A config runs on them through the pipeline's own functions, is scored the way the
organisers' scorer scores, and is compared with another config with the uncertainty that seven matches allow.
The plan is [P01](../../docs/proposals.md); what was measured is in [results.md](../../docs/results.md) (A05 to A09,
R002 to R004).

```
python pipeline/benchmark/sources.py          # data -> data/external/dfl/   (21 files, 2.6 GB, size and md5 checked)
python pipeline/benchmark/run.py CONFIG ...   # configs -> scores, intervals, comparisons, one JSON report per config
python pipeline/benchmark/fidelity.py         # how far the degraded profile looks like our match (gates, warnings)
python pipeline/benchmark/selftest.py [--dfl] [--data]   # the checks
```

## Setup

```
.venv/Scripts/python -m pip install -r requirements-benchmark.txt    # the pipeline's pins + lxml==6.1.3 (macOS/Linux: .venv/bin/python)
.venv/Scripts/python pipeline/benchmark/sources.py                   # download, then check every size and md5
.venv/Scripts/python pipeline/benchmark/sources.py --verify          # only check what is on disk
```

`sources.py` streams each file to `<name>.part`, hashes it as it arrives and renames it into place only if its size and
md5 are the pinned ones, so a file on disk is either complete and right or absent (a failed check is retried once from a
Hugging Face mirror of the same data). The files live in `data/external/dfl/<match id>/`, which git ignores; they are
public data and not the competition's, so unlike everything else under `data/` they may be read and printed.

What to know about the files (A05; `dfl.py` has the details and refuses what it cannot be sure of):

- A match is three XML files: the match information, the raw events (not read yet) and the observed positions
  (350 to 420 MB each). `dfl.load_match(match_id)` is the one door to a match: it parses the positions once, stores the
  arrays under `data/cache/dfl/<match id>/` (130 to 140 MB per match) and reads them back afterwards.
- **A first parse takes about 46 s** (21 to 64 s measured: the machine's load decides) and peaks at about 310 MB;
  **a cached load takes 0.7 to 1.9 s**, nearly all of it the sha256 of the two source files, which is what keeps a cache from being read after
  its sources change. The cache is keyed on those hashes, on the numpy and lxml versions, and on the hash of `dfl.py`
  itself, so **editing `dfl.py` rebuilds all seven caches (minutes)**.
- One timeline per match: the first half, 25 separator rows, then the second half. A row's place comes from the frame
  number `N`, never from the order of the file. Home and away come from each team's `Role`, never from the order of the
  teams in the file (the guest is listed first in four of the seven).
- `OtherGameInformation@TotalTimeFirstHalf` and `TotalTimeSecondHalf` are in **milliseconds**.
- Every squad member has a column in the position arrays, so **bench players are all-NaN columns** (7 to 11 of the 39
  to 40 per match). `nearest_player` already ignores an absent slot.
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
| `--profile P` | how the tracking is made (below): `gaps` (the default), `clean` or `tracking`. |
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

### The two truths

`--truth raw` is the DFL flags as they are: `D` where the ball is dead, otherwise the team in possession. DFL's flag is
fine-grained, about twice our ground truth's grain (A06), and how fine ours is cannot be told from the public window. So
there is a second truth, `--truth coarse` (`truth.coarse_codes`): the raw truth with every possession shorter than 5 s
that sits between two possessions of the *other* team absorbed into it, a stand-in for a coarser grain. **Run the
yardstick under both and read the two side by side**: a ranking that differs between them is a warning that the choice
depends on the grain of the truth, not something to fix. The two reports are filed apart (`<stem>.json` and
`<stem>.coarse.json`).

### The profiles

A profile says how the tracking of a DFL match is made before the config sees it (`degrade.py`). A profile has a version,
which goes into every report: change what a profile does and bump it.

| profile | version | the ball | the players |
|---|---|---|---|
| `clean` | 1 | DFL's, as it is: on every frame | DFL's, one slot per player |
| `gaps` | 1 | missing where **ours** was: our own missing-ball masks, transplanted onto each half; its speed is NaN everywhere (T31) | untouched |
| `tracking` | 1 | `gaps`'s, byte for byte | broken into tracklets of ours and given 46 slots, 24 home and 22 away, as ours are; a wrong-team slot now and then (10.5% of frames), NaN speeds in runs (7.8% of the slot-frames), 5-frame smoothing. **No frames with few players tracked yet** (below) |

**The transplant.** `Calibration.ball_masks[half]` is our match's missing-ball mask of each half from its kick-off to the
end of its graded block (`calibration.py`; the Calibration is measured once and cached). A benchmark half of n frames
takes the mask of our first or second half (1 or 2 with the same chance), read from an offset drawn uniformly over the
frames that leave room for it, so frame j of the benchmark half is missing where frame `offset + j` of ours was. If the
mask runs out, it goes on round the last 7,500 frames of ours. What it keeps of ours: the gaps' lengths, how much of the
half is missing and how that grows through the half (coverage worsens, A02). What it does not: **where** the gaps are.
They are drawn without looking at the play, so nothing that ties ours to it is in them: a dead ball is no likelier to be
missing than any other frame (in the public window ours is about three times as likely as a frame of the window in
general: 33.6% of the dead frames against 11.8% of all of them, A01), and where on the pitch a gap starts is the DFL
ball's own place at a random frame (36.0% of the onsets within 2 m of a line, against 28.0% of ours, R003). The
warnings and statistics of the fidelity report show how much that matters.

**The tracklets (`tracking`).** DFL has one column per player. Ours has 46 anonymous slots for 22 players, and a slot is
a tracklet: it ends when the tracker loses the player and is taken over by whoever it finds next (T15). The profile
breaks each player's time on the pitch (each contiguous run of frames where DFL has a position) into tracked spans and
lost spans. A tracked span is one of **our own tracklet lengths** (`Calibration.tracklet_lengths_s`, at the match's
rate, at least 1 frame), drawn uniformly; a lost span is geometric (at least 1 frame) with the mean that makes the lost
share d: d is 1 minus the mean number of slots present over the frames of ours with more than 18 present, over 22, which
is 0.0250, the mean tracked span 2,797 frames (111.9 s) and the mean lost span 71.7 frames (2.87 s) (A09). Frames with
18 or fewer present, 10.6% of ours, are the profile's tail, which is another part's and is left out so as not to count
it twice. All draws of the walk come from `rng_for(seed, match_index, "tracklets", half)`, players by column and runs in
time order.

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
which onsets are skipped. Over the seven matches and three seeds of R004, 5,330 onsets made 3,445 episodes (64.6%) and
skipped 1,759 as far (33.0%) and 126 as overlaps (2.4%). **s is 2.658, rescaled once** (it was 1.0): it calibrates the
observable share, which `calibration.player_stats` measures as ours was, and which is lower than the share of frames in
an episode (a team shows a surplus only while the other team has all 11 tracked and no episode of the other team runs
the other way at the same time). At s = 1.0 the observable share was 0.0397 against our 0.1054 +-0.03, so s became 1.0 *
0.1054 / 0.0397 and was frozen (`degrade.WRONG_TEAM_SCALE`, with the measurement in its comment; R004). It is a
label-free statistic and is never touched again because of a WARN or a score.

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
smoothed (`degrade.SMOOTH_WINDOW`). It lowers the pooled jitter to 6.47 cm (fidelity report, three seeds), still 20
times ours, and that is not the average's failure. Five of the seven matches (all the 2. Bundesliga ones) have 98% of
the squares of their second differences in the largest 0.05% of triples: rare jumps (in J03WOY's first half, 61 steps of
1,520,419 exceed 25 m/s) that an average spreads out and does not remove. And the bulk is a 1 cm grid: the median second
difference of every match is 1.00 cm and its 99th percentile 2.83 cm, against 1.4 mm and 9.6 mm in ours, which a 5-frame
average lowers by a factor of about 6 on white noise. Nothing was tuned to close that gap (R004).

**What this profile does not have yet.** Our frames with few players tracked (the tail: 10.6% of ours have 18 or fewer
present, 5.2% have 14 or fewer), which the next task adds as version 2 of the profile. Version 1 is **frozen as it is**,
and its three gates about how many slots are present, the share with exactly 22, the mean number and the share with 14
or fewer, are **pending**: printed with their numbers, not counted by the exit code, and listed in one constant
(`fidelity.PENDING_GATES`, with the reason for each), which the task that builds version 2 empties for it. Without the
tail too many frames have all 22 slots present: measured on full-strength frames (as it has to be) the share is 0.5631
against ours 0.5019 +-0.06, 0.0012 over the ceiling (R004), and the mean is 21.43 against 20.52 +-0.5 and the share with
14 or fewer 0.0000 against 0.0525 +-0.02. If the tail's cap is placed by our counts, independently of the benchmark's
own, each upper bucket loses about 10.6% of its frames and the first share becomes 0.5631 x (1 - 0.106) = 0.503 (an
estimate, not a measurement). The checks that pin the rest: `test_wrong_team_starts_at_close_pairs`,
`test_wrong_team_moves_one_player_to_other_pool`,
`test_wrong_team_draws_come_from_the_wrong_team_stream_in_a_fixed_order`, `test_speed_nan_share`,
`test_speed_nan_runs_follow_the_documented_draws`, `test_smoothing_is_a_centred_moving_average_inside_each_piece`,
`test_profiles_nest`, `test_tracking_without_the_new_parts_is_task_12s_players` (with the new parts off the players are
Task 12's, byte for byte), and on the downloaded matches `test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches`
and `test_data_tracking_constants_follow_from_the_measurements` (the rule that gave s and the window, applied again).
The earlier ones stay: `test_tracklets_never_overlap_in_a_slot`,
`test_each_tracklet_takes_the_lowest_free_slot_of_its_pool`, `test_slot_team_matches_pool`,
`test_tracking_degenerate_equals_gaps` (one endless tracklet per stretch of a player is `gaps`'s players),
`test_lost_share_on_long_synthetic`.

**No truth reaches the degrader.** It is handed a `TrackingView`, which has the tracking and the match's scalars and no
`ball_status` or `ball_possession` (`test_degrade_never_sees_truth`, which also reads the degrader's source, and the deep
walk of the inputs, `test_degrade_serves_inputs_without_the_truth`). Its random draws are
`rng_for(seed, match_index, component, half)`, one stream for each, with no profile and no loop order in the key
(`test_degrade_independent_of_match_order`), so a match gets the same degradation whichever matches are run with it, and
the `tracking` profile, which degrades the players as well, has exactly this profile's ball.

**A new calibration, a new report.** The report holds the digest of the Calibration its gaps were taken from. A report is
named by what was run (below), not by which calibration, so a later run with a newer Calibration writes over the older
report of the same name; compare the digests before comparing two reports.

### Fidelity: does the benchmark look like our match?

```
.venv/Scripts/python pipeline/benchmark/fidelity.py --profile gaps --seeds 3       # -> data/runs/benchmark/fidelity_gaps.json
```

`fidelity.py` degrades each match under seeds 0 .. N-1, measures it with **the same two functions that measured ours**
(`calibration.ball_stats` and `player_stats`), pools over matches and seeds (counts add, arrays are laid end to end: a
median is the median of every gap), and sets the result against the Calibration. Two kinds of criteria, fixed before
anything was measured and never tuned towards:

- **Gates**: the profile is not frozen until they pass, and the command exits 1 if one fails. Each is our Calibration's
  own value, measured on the kick-off spans (a benchmark half starts at its kick-off, so both sides are measured alike),
  plus or minus the plan's tolerance. For `gaps`: the share of frames without a ball, +-0.02, and the median gap, +-1.5
  s. For `tracking`, those two and: the share of frames with exactly 22 slots present +-0.06, the mean number present
  +-0.5, the share with 14 or fewer present +-0.02, the wrong-team share +-0.03, the tracklet median **+-50% of the
  centre** (a relative tolerance; the rest are absolute) and the NaN-speed share +-0.01. **The three gates on how many
  slots are present are measured on full-strength frames only**: those where the clean DFL match has exactly 22 players
  with positions, because a team that is down to 10 cannot show 22 slots whatever the degradation does (two of the seven
  matches have one, from minute 7 of the first half of J03WN1 and minute 13 of the second half of J03WQQ, and 18.3% of
  the pooled frames have 21 players: 81.7% are at full strength), and ours is measured on all its frames, which assumes
  that it has 22 players on the pitch throughout. Every other gate and every warning stays on all frames; the report
  holds the count statistics on both sets of frames, how many frames are at full strength and which matches are not all
  of them (`full_strength`). The plan's own figures (0.407, 10.7 s, 0.108, 35.5 s, ...) were measured on other spans and
  are not used. **Pending gates** (`fidelity.PENDING_GATES`) are printed and not counted: today the three about how many
  slots are present (exactly 22, the mean, 14 or fewer), which depend on the low-count tail that `tracking` does not
  have yet ("What this profile does not have yet", above, says how). A pending gate is not counted, so version 1 is
  frozen with those three pending, and the tail is version 2.
- **Warnings**: the nearest player's agreement with the truth on alive, ball-visible frames (our public window's, +-0.05),
  the share of dead frames with the ball visible (+-0.10), and for each yardstick its possessions per graded minute
  (+-25%) and its D share (+-0.05), against what the yardsticks output on ours (over our graded halves, as the published
  baselines are: an accepted offset of at most 1% of the frames). A WARN is printed and discussed in `results.md`; the
  degradation is not adjusted towards it, because some misses are the truth's conventions and not the degradation's (A06).

The report also lists the benchmark's other statistics next to ours (tracklets, wrong-team frames, NaN speeds,
jitter...), which `gaps` does not model, so that what a later profile has to fix is on the page, the degrader's draws
(two integers a half) and, for `tracking`, the pooled counts of the players' draws (onsets, episodes, skips, NaN-speed
runs). It holds no frame of our match, and it carries the Calibration's digest. It is named
`fidelity_<profile>[.m-<ids>][.s<n>].json` by the runner's rule.

### What the clean profile cannot show

The DFL ball is tracked on every frame, so `nearest_player` votes everywhere and `majority_vote`, which takes `D` only
where nobody voted, labels almost nothing `D`, where the truth has 41% to 52% of every match dead. So on `clean`
the alive/dead F1 is the same for every yardstick (it is the F1 of "always alive", the scorer's ALIVE being the positive
class), the macroF1 is held down by an empty `D` class, and the ranking is only about who has the ball. That is
expected, and is why the degraded profiles (which give the ball our tracking's gaps) exist. It is not a bug of the
configs. `gaps` gives the ball our gaps, so that `majority_vote` has `D` to give, and the scores are no longer those of a
ball that is always there.

## Reading the printout

Per config:

- **pooled** spellF1, macroF1 and alive/dead F1: the official scorer's numbers for all the halves of all the matches as
  one file. They are computed from summed counts, never from an average of averages, and `score.py`'s checks assert that
  they equal `pm.evaluate` on the halves laid end to end.
- **per match**: the mean of the seven matches' own spellF1 and its 90% Student t interval. This one is not the
  scorer's number: it weighs each match once, and it is the one an interval can be put on.
- **possessions** per graded minute, predicted and true, and the share of graded frames labelled `D`, predicted and true.
  A config that flickers shows up here before it shows up in a score.
- spellF1 by match, and the seconds the config took (printed, never written to a report).

With two or more configs, each against the first:

- **pooled delta** and **mean per-match delta**, with the 90% t interval of the match deltas. An interval is said to
  **exclude 0** only if it clears 0 by more than `score.FLIP_TOLERANCE` (1e-12, the margin under which two scores are
  equal), because a delta of the order of 1e-16 can otherwise make a gap out of float noise.
- **sign-flip p**: the exact two-sided randomisation test of the seven deltas, with all 2^7 sign patterns enumerated. The
  smallest p it can give with seven matches is 2/128 = 0.0156, and a config that wins every match gets exactly that.
- **wins and losses**: the matches whose delta is above and below 0 by more than `score.FLIP_TOLERANCE`.
- **by competition** and **per match**: the same delta for the two Bundesliga matches and the five 2. Bundesliga ones,
  and for each match. Five of the seven are Fortuna Düsseldorf home games, so seven matches are not seven independent
  draws of "a match"; a gap that holds in every match and in both competitions is worth more than one that does not.

The uncertainty is **never a bootstrap**: resampling seven matches, or the frames inside them, would pretend to a sample
the benchmark does not have. An interval says how far to trust a gap; it is not a licence to prefer a config on a gap it
does not clear, and the public window is no tie-breaker either (T01).

### The report

`<data dir>/runs/benchmark/<profile>/<stem>[.<truth>][.m-<ids>][.s<n>].json`, one per config, gitignored with the rest of
`data/`. The name says what the report is of, so that one never replaces another of a different kind:

| run | file |
|---|---|
| all seven matches, raw truth, the default 3 seeds | `<stem>.json` |
| all seven, coarse truth | `<stem>.coarse.json` |
| any other set of matches | `.m-<the ids, sorted, joined by "-">` before `.json`, e.g. `<stem>.m-J03WMX.json`, `<stem>.coarse.m-J03WMX-J03WN1.json`. The order they were asked for in is not in the name; all seven in another order are still `<stem>.json` |
| a number of seeds other than 3 (a degraded profile) | `.s<n>` last, e.g. `<stem>.s1.json`, `<stem>.coarse.m-J03WMX.s5.json`: a quick one-seed look never replaces the three-seed report. The clean profile has one seed, always, and its name never says so |

A config's own file name may hold no dot: `x.coarse.toml` would write `x.coarse.json`, which is `x`'s report under another
truth, so the runner refuses it and says to rename it.

It is written with sorted keys, indented, LF, floats rounded to 9 decimals and no timing anywhere, to a temporary file
renamed over the old report; the same inputs, seed and versions give the same bytes in any process.

| key | holds |
|---|---|
| `config` | `path`, and `text`: the TOML as run, with LF line endings |
| `profile`, `seeds`, `truth` | the profile's name and version, the seeds run, and `{name, min_s}` of the truth |
| `calibration` | `{sha256, version}` of the Calibration the tracking was degraded with (`calibration.digest`); null for `clean`, which degrades nothing |
| `matches` | per match, in the order run: `id`, `frames`, `graded_frames`, `fps`, `carried_frames` (frames the truth had to carry: 0), and `sources`: the size and sha256 of the two DFL files it was made from (null if the match was not read through the cache, with `sources_reason`) |
| `git`, `environment` | the commit and whether the tree was dirty; the Python, numpy and lxml versions and the platform |
| `pooled`, `per_match` | the scores of all halves, and of each match: `spell_f1_mean`, `spell_f1` at each IoU, `macro_f1`, `alive_dead_f1`, `n_gt`, `n_pred`, `halves` |
| `t_interval` | the t interval of the per-match `spell_f1_mean` and `macro_f1`: `n` (the matches), `level` (0.9), `mean`, `low`, `high`. Only `low` and `high` are null, for fewer than two matches; the `mean` of one match is its own score |
| `possessions_per_graded_minute`, `labels` | the rates and the label shares of the printout |
| `halves` | the counts everything above was made from, per half: `n_gt`, `n_pred`, `tp` at each IoU, the 3x4 `confusion` of the unfilled codes |
| `versus` | null for the reference; for the others, the comparison with it for `spell_f1_mean` and `macro_f1` (`score.compare`: deltas, interval, p, wins, losses, by competition, and whether the interval excludes 0) and the reference's name and config hash. It is the one field that depends on which other configs were run |

## The checks

```
.venv/Scripts/python pipeline/benchmark/selftest.py          # synthetic, seconds
.venv/Scripts/python pipeline/benchmark/selftest.py --dfl    # + the seven downloaded matches: 99 to 221 s measured with the caches warm
```

(The first time, while the seven caches are built, the `--dfl` run took 532 s; the machine's load decides the rest.)

`checks/run.py` holds the runner's, `checks/degrade.py` the degrader's shared parts (the transplant worked out exactly,
determinism, what the `gaps` profile touches and what it does not, no truth, the order of the matches),
`checks/tracking.py` the `tracking` profile's (the tracklets, the episodes, the NaN-speed runs and the smoothing worked
out exactly, and how it nests in `gaps`) and `checks/fidelity.py` the fidelity report's (every gate and check worked out
by other code, pooling by sums, the exit code, the pending gates, the full-strength frames of the count gates and the
gate kinds). The
one that anchors the rest is `test_dfl_oracle_scores_perfectly`: an evidence source that is the truth itself, decoded by
`majority_vote` with half-width 0, must score exactly 1.0 on spellF1, macroF1 and alive/dead F1 on every one of the
seven matches.

**What the oracle checks, and what it cannot.** It checks the chain from the *loaded match* to the score: the runner,
`MatchInputs` and the separator rows between the halves, both halves and each one's first and last row, the decoder path
and the scoring. A row too many or too few at the half time, a lost first row, a half decoded one row late or a team
swapped on the way would not score 1.0. It does **not** check the link from the DFL file to the truth: the oracle's
answers and the runner's truth are the same `truth.truth_codes` of the same loaded match, and the oracle never reads
the tracking, so a mistake in the parse or in `truth_codes` is on both sides and cancels (with H and A swapped inside
`truth_codes` it still scores 1.0 on all seven matches). That link is covered by A05's checks of the parse against
kloppy and by frame number (`test_dfl_parser_agrees_with_kloppy`, `test_dfl_second_half_spot_check`), and for the teams
by `MIN_AGREEMENT` in `test_dfl_truth_conventions`: the nearest player must name the truth's team on more than 60% of the
alive, ball-visible frames of every match and of each half (it does, at 71% to 82%; with the teams swapped it would be
at 18% to 29%).

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

*A stub: the final version is written once the remaining profiles exist.*

- Seven matches, five of them one club's home games, are a small and not independent sample; the intervals are wide for
  that reason and are no more than that.
- DFL's flags and Metrica's corrected tracking are different annotation systems, and DFL's grain is finer than ours as
  far as we can tell (A06). That is what `--truth coarse` is for, and it does not correct a measured difference.
- The benchmark has no video and no tagging: whatever uses the homography, the Smart Tagging, the camera or the audio is
  checked on the public window and not here.
- The clean profile has the ball on every frame, and says how a method works when it can see the ball. The `gaps` profile
  has our gaps and nothing else of our tracking: its players are DFL's (one slot per player, no wrong-team slot, no
  NaN speeds), and its gaps are not placed by the play. The warnings and statistics of the fidelity report say by how
  much.
- The `tracking` profile has our tracklets, slots, wrong-team episodes, NaN-speed runs and smoothing, and **not our
  frames with few players tracked** (the tail, which the next task adds; three of its gates, the ones about how many
  slots are present, are pending until then). Its slots are the lowest free ones, so a team uses 13 of its slots over a
  whole match (J03WMX, seed 0: 11 home and 12 away slots are present on at least a tenth of the frames) where ours has
  21 of 24 and 20 of 22 present on at least a tenth of the frames, and a slot's index is not as empty of the player as
  in ours: a player who comes back after a lost span lands in the slot he had 69.0% of the time (860 of 1,247 returns,
  J03WMX, seed 0, with the episodes on, as in version 1 of the profile), and a slot idles between two tracklets for a
  median of 2.3 s (59.7% of its gaps at most 3 s, 3.8% over 60 s), where ours idles for 7.4 s (17.7% and 15.0%; A09,
  R004). Before the wrong-team episodes were added the figures were 76.0%, 2.0 s, 67.1% and none over 60 s (A09; the
  same code with the scale set to 0 gives them again). No evidence source reads a slot's identity today, beyond
  `nearest_player`'s argmin on exact ties, which goes to the lower slot; but **a method that follows a slot across a
  loss would be validated optimistically here**, and the tail's part of the profile re-measures this. The episodes are
  cut at pairs within 2 m only, in both directions at one rate, and what is matched is the observable share, not the
  episodes: on seed 0 the benchmark shows 105 to 200 observed episodes a match (mean 163), medians of 1.4 to 2.2 s and
  90th percentiles of 6.2 to 9.4 s, against ours 104, 2.7 s and 15.8 s: more of them and shorter (a swap inside 11
  against 11 is invisible on ours, so the true number of ours is unknown).
- **No positional bias.** The profile matches the players' jitter and nothing else of their positional error. A slowly
  varying bias against the true position cannot be measured on ours (there is no reference to measure it against), so
  none is added. Published accuracies for video-derived positions run from 0.08 to 0.09 m RMSE for a multi-camera
  stadium system against a motion-capture reference (Linke, Link and Lames 2020, PLoS ONE 15(3): e0230179) to a median
  of 1.1 to 1.7 m (mean 1.8 to 2.8 m) for a pipeline on broadcast video (Theiner et al., WACV 2022, arXiv:2110.11107,
  Table 3), so a bias of the order of 0.5 to 2 m is plausible for ours and would hurt a method that reads distances more
  than the benchmark says (the plan's range of 0.5 to 2 m is not stated by either source, which are the nearest
  published figures; it is not our measurement). DFL's own positions are not noise-free either: their second difference
  is dominated by rare jumps (R004), which the smoothing leaves in.
- A transplanted mask is cut at the edge of the window it is read from, so up to two gaps of a benchmark half (the ones
  at its ends) are fragments of ours, and where the mask runs out and goes round there is one more at the seam (3 of the
  42 halves of R003: a DFL second half of 77,080 or 78,456 frames, against the 76,456 of ours); the gates are on the
  share and the median, which this barely moves.
