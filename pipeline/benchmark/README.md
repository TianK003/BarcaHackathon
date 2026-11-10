# The P01 benchmark

We may not hand-label our own match, not even to validate (the organisers' clarification of 22 Oct 2026,
[CLAUDE.md](../../CLAUDE.md)), and since their clarification of 3 Nov 2026 we may not train or tune on anything of it,
labels or not: "you MUST use only external data" ([docs/rules.md](../../docs/rules.md)). The public window holds 17 possessions and ranks the baselines the wrong way round
([T01](../../docs/traps.md)). So the choices that matter (decoder, evidence weights, thresholds) are validated on
**other matches**: the seven open DFL/IDSSE matches (two Bundesliga, five 2. Bundesliga; CC BY 4.0), which carry the
ball's own alive/dead flag and the team in possession on every frame. A config runs on them through the pipeline's own
functions, is scored the way the organisers' scorer scores, and is compared with another config with the uncertainty
that seven matches allow.
The plan is [P01](../../docs/proposals.md); what was measured is in [results.md](../../docs/results.md) (A05 to A11,
R002 to R012).

**Since R030 (10 Nov 2026) the benchmark's default profile is `generic`, which reads nothing of our match**
([P27](../../docs/proposals.md), [the spec](../../docs/superpowers/specs/2026-11-09-p27-degradation-design.md)). Both
models are trained on it and every numeric constant of `main.toml` is chosen on it. The three older degraded
profiles (`gaps`, `tracking`, `context`) read our match. Since the organisers' reply of 5 Nov nothing is trained or
chosen on them, and they stay for the record of R002 to R029 ([The profiles](#the-profiles)).

```
python pipeline/benchmark/sources.py          # data -> data/external/dfl/   (21 files, 2.6 GB, size and md5 checked)
python pipeline/benchmark/run.py CONFIG ...   # configs -> scores, intervals, comparisons, one JSON report per config;
                                              # --profile generic is the default since R030
python pipeline/benchmark/choose.py --reference main STEM ...   # a grid's declared rule applied to its reports, + LOMO
python pipeline/benchmark/choose.py --reference main --objective mean STEM ...   # R023's objective: the mean of the two truths (R024)
python pipeline/benchmark/p27_grid.py ref|a|c|d|final --confidences FILE ...   # P27 (R030): write a stage's configs into configs/p27/
python pipeline/benchmark/p27_choose.py a|c|d            # P27: the declared rule applied to a stage's reports; prints the next call
python pipeline/benchmark/p27_choose.py bar STEM         # P27: the final bar, STEM against the two yardsticks
python pipeline/benchmark/p27_choose.py guard STEM BASE  # P27: the shift's guard, STEM (shifted) against BASE (not)
python pipeline/benchmark/p27_shift.py STEM   # P27 Stage E: the shift, from the pipeline's own lag on DFL (generic, raw truth)
python pipeline/benchmark/p27_profile_check.py   # P27: the generic profile against its declared shares, on the 21 match-seeds
python pipeline/benchmark/diagnose.py CONFIG --drop-match-only   # A19: the IoU ladder decomposed, the sources' and the posterior's calibration
python pipeline/benchmark/fidelity.py --profile tracking|gaps|context   # how far a profile that reads our match looks like it
                                              # (gates, warnings); default tracking; generic has no gates, by design
python pipeline/benchmark/headroom.py CONFIG  # a config beside five variants whose evidence is the truth itself
python pipeline/benchmark/label_diff.py REFERENCE CONFIG ...  # how configs change BAR-ATH's labels beside a reference, aggregates only (heavy)
python pipeline/benchmark/tag_stats.py        # P08: Smart Tagging's votes against our own pseudo-labels (R017), aggregates only (heavy); since 3 Nov a check only, never a constant (docs/rules.md)
python pipeline/benchmark/wrong_team.py       # P21 step 0: wrong-team slots at the ball against all graded frames (A16), aggregates only (heavy)
python pipeline/benchmark/metrica.py [--offline]   # P16: the grain of H/A/D derived from Metrica's sample games' events (R023);
                                                   # not to be run since 5 Nov 2026: the data has no licence (docs/rules.md)
python pipeline/benchmark/touch_stats.py      # P04's control rule measured: coverage, agreement, flip lags, confidence
python pipeline/benchmark/train_dead_ball.py --model hgb   # P05: train the dead-ball model (folds + full) as JSON; --profile generic is the default
python pipeline/benchmark/train_shape.py [--verify|--report]   # P07: train the shape model (folds + full) as JSON; --profile generic is the default
python pipeline/benchmark/conventions.py public|dfl|compare|latency|shift ...   # P02: when a truth flips, the shift
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

**The restart events (`dfl_events.py`, P17's diagnostics).** The second of a match's three files, the raw events, names
every restart: KickOff, ThrowIn, CornerKick, GoalKick, FreeKick and Penalty, each with the team that takes it. They are
human-collected event data of these public matches (Bassek et al. 2025, CC BY 4.0: the paper says trained human operators
record them), about other games, and they are used for one diagnostic only, `restarts.py events` (nothing is trained,
fitted or chosen from them; the technical document's section 2 lists them with the labelled data). `dfl_events.load_restarts`
puts them on the rows of `dfl.py`'s timeline through the position file's own frame clock (a row is the half's start plus
the time since the half's first ball frame, in 40 ms steps, rounded), and refuses a file whose clock is not regular; a
restart outside the graded rows is dropped and counted. No pipeline component, decoder or choice reads it, and it does
not touch `dfl.py`, so the caches are not rebuilt.

## Running a config

```
.venv/Scripts/python pipeline/benchmark/run.py \
    pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w25.toml \
    pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/nearest_w0.toml
.venv/Scripts/python pipeline/benchmark/run.py <the same configs> --truth coarse
.venv/Scripts/python pipeline/benchmark/run.py <the same configs> --truth both     # the two runs, from one decode
```

| option | meaning |
|---|---|
| `CONFIG ...` | pipeline TOML configs (`pipeline/config.py`), per half (`scope = "halves"`). **The first one is the reference**: every other config is compared with it, and a delta is `other - first`, so positive means the other is better. Put the config you want to beat first. |
| `--profile P` | how the tracking is made (below): `generic` (the default since R030: it reads nothing of our match), `clean`, or one of the three that read our match and are kept for the record of R002 to R029 only: `tracking` (the default until R030), `gaps` and `context` (a sensitivity check only: it fails two of its gates, R005, and the help, the printout and the report say so). |
| `--seeds N` | degradation seeds per match (default 3: seeds 0, 1 and 2). The clean profile has nothing random to repeat, so it always runs seed 0 only; if you pass `--seeds` with more than 1 it says that it ignored it, and says nothing when you pass none. |
| `--matches ID ...` | the matches to run, in this order (default: all seven). Fewer than seven are filed under a name that says which (see the report, below). |
| `--truth raw\|coarse\|both` | the truth to score against (below). Default `raw`. `both` decodes each config once and scores the codes against each truth (`evaluate_truths`): the two reports and tables are exactly those of a raw and a coarse run (`test_runner_scores_both_truths_from_one_decode`), in about half the time, because the decoder and not the scoring is what a config's time goes on (R013). |
| `--data-dir D` | the data bundle (default `data/`, or `$MTAH_DATA_DIR`). |
| `--no-write` | print only. |
| `--drop-match-only` | drop, from every config, the evidence sources that declare `MATCH_ONLY` (below). |

A run takes about 10 s on the clean profile with the caches warm (one cached load per match, a second or so per config);
a degraded profile runs each config once for each of its seeds. Everything that can be refused is refused before the first
match is loaded, with a sentence that names what to change: no config, a config that does not load, `scope = "file"`
(which decodes across half time; only `pipeline/configs/smooth50_repro.toml` needs it), a config whose file name has a
dot, two configs with the same name, an unknown profile, truth or match, a calibration that is not one, and a
Calibration handed to `generic` or `clean`, which take none.
A source that needs something a benchmark match does not have (the video's homography, the Smart Tagging) stops the run
with its config, its name and kind and the asset named; it is never scored as zeros.
A source that needs an asset only our match has (the Smart Tagging, the whistle) declares `MATCH_ONLY = True` in its own
module (`evidence.match_only(kind)` reads it), and its error then ends with a hint to run with `--drop-match-only`. That
flag runs each config without such sources (`run.drop_match_only`), never silently: the printed block names what was
dropped, and every report carries `dropped_match_only` (a list of source names, `[]` when nothing was dropped; a report
from before the flag has no such key, and `choose.py` never reads it). The report still quotes the config as written,
so its numbers are those of the config without those sources and say so only through that field. `conventions.py
latency CONFIG dfl` takes the same flag.

### The yardsticks

`configs/nearest_w{0,25,50,125}.toml` are the organisers' four published baselines (`baseline_raw`, `baseline_smooth25`,
`baseline_smooth50`, `baseline_smooth125`: the nearest tracked player's team on every frame, then a centred majority
vote of +/- w frames) **decoded per graded half**, as every config of ours is. The published files are decoded over the
whole file at once, so these are yardsticks for the benchmark and not rebuilds of them (that is
`pipeline/configs/smooth50_repro.toml`). A change has to beat them here before it is worth anything.

P06's grid configs live in `configs/p06/`, named `p06s1_t{TTT}_d{DDD}.toml` for Stage 1: the hmm decoder with
`switch_team` TTT and `switch_dead` DDD (zero-padded to three digits), `nearest_player` at confidence 0.78 and
`ball_presence` at its defaults. The 24 files cross `switch_dead` d in {4, 8, 16, 32, 64, 128} with `switch_team` in
{d/4, d/2, d, 2d} (R006). Stage 2 is `p06s2_t{TTT}_d{DDD}_mp{MMM}_md{NNN}.toml` (90 files, with `min_possession` MMM and
`min_dead` NNN added), Stage 2b `p06s2b_t{TTT}_d{DDD}_mp{MMM}_md001.toml` (15 files) and the four controls
`p06ctl_{seen_off,margin50,d008,d128}.toml`, each the chosen `pipeline/configs/p06_hmm.toml` with one thing changed
(R007).

### The two truths

`--truth raw` is the DFL flags as they are: `D` where the ball is dead, otherwise the team in possession. DFL's flag is
fine-grained (437 possessions a match, a median of 3.7 s: A06), and how fine our ground truth is cannot be told from
the public window (A06, T33). So there is a second truth, `--truth coarse` (`truth.coarse_codes`): the raw truth with
every possession shorter than 5 s that sits between two possessions of the *other* team absorbed into it, a stand-in
for a coarser grain (1,659 possessions against 3,061, A06). **Run the yardstick under both and read the two side by
side**: a ranking that differs between them is a warning that the choice depends on the grain of the truth, not
something to fix. The two reports are filed apart (`<stem>.json` and `<stem>.coarse.json`).

Which one our ground truth resembles was measured on other games of its provider. `metrica.py` (P16,
[R023](../../docs/results.md)) fetches Metrica's three public sample games' event files: pinned to one commit, checked
by git blob SHA, and never committed. It derives H/A/D from them by a rule declared before the fetch and compares their
grain with A06's two truths: 6.20 possessions per alive minute, between 8.30 (raw) and 4.50 (coarse), and a median of
4.72 s, on the raw side. The outcome was **both**: a choice declared after R023 ranks on the unweighted mean of the
two pooled spellF1s, each reported. It is event-tagged possession, not a tracking flag, so how far that grain sits
from our flag's is not measured (R023's stated limit). **Since 5 Nov 2026 the sample data is withdrawn:** it has no
licence, and the organisers' reply ("any dataset you have the rights to use", "anything that is not explicitly allowed
is not permitted") leaves us without the rights, so `metrica.py` is not run again. Tian decided on 9 Nov that R024's
objective stays, on A06's ground alone (the grain of our truth is unknown, so neither truth is preferred), and R030
re-chose every constant under it ([docs/rules.md](../../docs/rules.md), question 3).

### The profiles

A profile says how the tracking of a DFL match is made before the config sees it (`degrade.py`). A profile has a
version, which goes into every report: change what a profile does and bump it. There are five, of two kinds.

**`generic` (version 1) is the default since R030** (10 Nov 2026; `run.DEFAULT_PROFILE = "generic"`). It reads nothing
of our match. Every setting is a figure that the organisers publish in DATA.md or a round rule declared before any
run ([the spec](../../docs/superpowers/specs/2026-11-09-p27-degradation-design.md)), and no `Calibration` is loaded
for it (`degrade.needs_calibration("generic")` is False). It is the only degraded profile that anything is trained
or chosen on. Version 1 is final for this submission: a version 2 is made only on a rules ground, never after a
score. `clean` reads nothing of our match either, and degrades nothing: it is the floor.

**`gaps`, `tracking` and `context` read our match, and are retired for training and choosing**
(`degrade.NEEDS_CALIBRATION = {gaps, tracking, context}`). Their constants were fitted once per version, from
label-free measurements of our tracking, and then frozen with the measurement in the code; they were never tuned
towards a gate, a WARN or a score (T26, T37). No label, ground truth or video of our match entered them, and every
choice made on them was made on DFL's own labels. Whether fitting a profile to our tracking is allowed under the
organisers' clarification of 3 Nov 2026 (training and tuning on external data only) was question 1 for the organisers in
[docs/rules.md](../../docs/rules.md). They replied on 5 Nov with the clarification and "anything that is not explicitly
allowed is not permitted", so by our reading it is not permitted. Tian decided the redo on 9 Nov (P27 in
[docs/proposals.md](../../docs/proposals.md)), and R030 did it on 10 Nov. The three profiles stay in the code for the
record of R002 to R029, and nothing is trained or chosen on them again.

| profile | version | the ball | the players |
|---|---|---|---|
| `generic` | 1 | missing on 41% of the frames in expectation (DATA.md's share of the graded frames), in gaps whose length is log-uniform between 1 s and 60 s and whose onsets are a Poisson process blind to the play; its speed is NaN everywhere | each on-pitch player is lost on 1 − 20.5 / 22 = 6.8% of the frames (DATA.md's mean of 20.5 tracked objects), independently of the others, in tracked spans log-uniform between 2 s and 300 s; 46 slots, 24 home and 22 away, each tracklet in the lowest free slot of its pool; a centred 7-frame moving average over each tracked piece; no wrong-team slots, no NaN-speed runs (a speed is NaN only on a tracklet's first row), no low-count frames and no shortest piece. **The default since R030** (below) |
| `clean` | 1 | DFL's, as it is: on every frame | DFL's, one slot per player |
| `gaps` | 1 | missing where **ours** was: our own missing-ball masks, transplanted onto each half with their pattern and not their clock (below); its speed is NaN everywhere (T31) | untouched |
| `tracking` | 3 | `gaps`'s, byte for byte | broken into tracklets of ours and given 46 slots, 24 home and 22 away, as ours are; a wrong-team slot now and then (10.5% of frames), NaN speeds in runs (7.8% of the slot-frames), 5-frame smoothing; **since version 2** only the players nearest the ball are tracked where ours has few (the cap), no piece is shorter than 8 frames, and the walk's spans are rescaled once (below). **since version 3** a slot is taken by a new tracklet only from the row after its last one stopped, and the scale is derived again (below); this was the default profile from R004c until R030. Version 1 (R004) had none of the three of version 2, and versions 1 and 2 had the slot handovers that version 3 removes (R004c) |
| `context` | 1 | missing where our tracker loses it: a hazard fitted on our own match (A08) places the gaps by the ball's distance to a line, its speed, the players within 3 m of it (a crowd raises the hazard) and the minute of the half, with our gap lengths (`HAZARD_SCALE` 0.699, rescaled once). **It fails two of its gates**, the near-line share badly (its gaps start within 2 m of a line 56% of the time, against 28% of ours) and the missing share narrowly (R005), so it is **not the default**, and is kept for the agreement check between generators | `tracking`'s, draw for draw (`test_context_players_equal_tracking`) |

**The `generic` profile** (version 1; `degrade.GENERIC_*`, `generic_mask`, `generic_parameters`). Three sources may
shape it, and nothing else: a figure the organisers publish in the challenge documents, a rule declared in the spec
before any run, and the file format the pipeline parses. What we measured on our match (A01 to A03, A07 to A09, the
Calibration) may not, nor may a number picked because it lands near one of those.

- *The ball.* Each half's mask runs from its kick-off to the end of its graded block and is drawn from the half's
  own stream, `rng_for(seed, match_index, "generic_gaps", half)`: missing and visible spans in turn. A missing span
  is log-uniform between 1 s and 60 s (`GENERIC_GAP_SECONDS`): every scale from a ball in the air to a long stoppage
  is equally likely. A visible span is exponential, so the gaps start as a Poisson process, with the mean that makes
  the expected missing share 0.41 (`GENERIC_MISSING_SHARE`; DATA.md: "The ball is missing 43% of all frames, and 41%
  of the graded ones"). Nothing of the play is read. The ball's speed is NaN on every frame, so no component can lean
  on one. By arithmetic the law gives a median gap of 7.7 s, a mean of 14.4 s and a 90th percentile of 40 s.
- *The players.* `tracking`'s machinery with declared parameters. An on-pitch player is lost on d = 1 − 20.5 / 22 of
  the frames (`GENERIC_PRESENT_MEAN`; DATA.md: "the mean over graded frames is 20.5"), independently of the others. A
  tracked span is log-uniform between 2 s and 300 s (`GENERIC_TRACKED_SECONDS`), and a lost span is geometric with the
  mean that makes the lost share d. The pieces take the lowest free slot of 24 home and 22 away slots and are smoothed
  with a centred 7-frame moving average (`GENERIC_SMOOTH_WINDOW`, the window of Shaw's velocity filter for 25 Hz
  tracking). There are no wrong-team episodes, no NaN-speed runs beyond a tracklet's first row, no cap and no shortest
  piece: DATA.md documents none of them.
- *No Calibration, by code.* `degrade.degrade` refuses one for `generic`, and so do the runner and the trainers. A
  report's `calibration` is null, and so is `calibration_sha256` in a model's provenance.
- *Two figures are quoted from the task statement,* as 25 fps and 46 slots are: the 41% and the 20.5. They are
  statistics of the match, published by the organisers. No text allows quoting them by name, so a residual risk
  remains ([docs/rules.md](../../docs/rules.md)).
- *A check made after the law was written, on external data only.* In SkillCorner's open data (MIT licence, automatic
  broadcast tracking, three A-League matches read) the ball is undetected on 39% to 46% of the frames, in gaps with a
  median of 5.4 to 5.8 s, a mean of 11.5 to 14.0 s and a 90th percentile of 27 to 36 s. No number was taken from it.
- *The check of the profile itself* (`p27_profile_check.py`: the 21 match-seeds degraded again, counts only). The
  pooled share of the ball's masks is 0.4109 against the declared 0.41 ± 0.02, and the pooled share of on-pitch
  player-frames lost is 0.0681 against 0.0682 ± 0.01. Both pass (R030). A miss would be a bug in the code.
- *No fidelity gate against our match.* `fidelity.py` measures a profile against the Calibration. `generic` is not
  built to pass it and is never run through it.

Its checks are in `checks/generic.py` (synthetic): `test_generic_constants_are_the_declared_ones`,
`test_generic_takes_no_calibration_and_refuses_one`, `test_generic_no_instrument_loads_a_calibration`,
`test_generic_is_held_to_no_fidelity_gate_against_our_match`, `test_generic_never_sees_the_truth`,
`test_generic_is_deterministic_and_independent_of_the_other_matches`, `test_generic_mask_is_the_documented_draws`,
`test_generic_ball_is_missing_exactly_where_the_masks_say`, `test_generic_parameters_are_the_declared_rules`,
`test_generic_players_are_lost_at_the_declared_rate_in_46_slots` and
`test_generic_runs_on_the_runner_with_no_calibration`.

*From here to "No truth reaches the degrader" the text describes the three profiles that read our match. It is the
record of how R002 to R029 were measured. Nothing is trained or chosen on those profiles since R030.*

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
m** (identity swaps happen where players come close) does a random one of the two become an episode: their rows from the
onset to the end of the episode (or of their tracked span) form a separate tracklet in **the other team's pool**, and the
rest of their span is cut into a piece before it and a new tracklet after it (the split is an identity switch). An onset
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
lowest free slot, so a player who comes back after a drop may not get their own slot back (the first thing the cap does to
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
the `tracking` profile, which degrades the players as well, has exactly `gaps`'s ball. The same holds for `generic`
(`test_generic_never_sees_the_truth`, `test_generic_is_deterministic_and_independent_of_the_other_matches`), whose
ball has its own stream, `generic_gaps`.

**A new calibration, a new report.** For `gaps`, `tracking` and `context` the report holds the digest of the
Calibration its gaps were taken from (for `generic` and `clean`, which take none, it holds null). A report
is named by what was run (below), not by which calibration, so a later run with a newer Calibration writes over the
older report of the same name; compare the digests before comparing two reports.

### Fidelity: does the benchmark look like our match?

```
# -> data/runs/benchmark/fidelity_tracking.json
.venv/Scripts/python pipeline/benchmark/fidelity.py --profile tracking --seeds 3
```

**Fidelity is measured for `gaps`, `tracking` and `context` only** (`fidelity.GATES`): the profiles that read our
match. `generic` has no gates, by design: it is not built to look like our match by measurement, and `fidelity.py
--profile generic` is refused before a Calibration or a match is loaded
(`test_generic_is_held_to_no_fidelity_gate_against_our_match`). A bare `fidelity.py` measures `tracking`
(`fidelity.DEFAULT_PROFILE`), not the runner's default. Since R030 a fidelity report is part of the record of R003 to
R005. It gates nothing that is trained or chosen today, and no profile may be changed to pass it (T01).

`fidelity.py` degrades each match under seeds 0 .. N-1, measures it with **the same two functions that measured ours**
(`calibration.ball_stats` and `player_stats`), pools over matches and seeds (counts add, arrays are laid end to end: a
median is the median of every gap), and sets the result against the Calibration. Two kinds of criteria, fixed before
anything was measured and never tuned towards:

- **Gates**: the command exits 1 if one fails, and until R030 a profile could be the default only if all its gates
  passed (`context` fails two and is kept as a sensitivity check, R005; `generic`, the default since R030, has none).
  Each is our Calibration's
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
  degradation's (A06). The first two set the copy beside statistics of the public window's labels, as a check on the
  copy and nothing more (the window is "not a tuning set", RULES.md, 23 Oct 2026): no profile constant is fitted to
  them.

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
expected, and is why the degraded profiles (which take the ball away in gaps) exist. It is not a bug of the
configs. `generic` draws the ball's gaps from its declared law, and `gaps` gave the ball our gaps, so that
`majority_vote` has `D` to give, and the scores are no longer those of a ball that is always there.

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
| `calibration` | `{sha256, version}` of the Calibration the tracking was degraded with (`calibration.digest`), for `gaps`, `tracking` and `context`; null for `generic` and `clean`, which take none |
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
.venv/Scripts/python pipeline/benchmark/headroom.py CONFIG [--profile P] [--seeds N] [--truth raw|coarse] [--matches ID ...] [--data-dir D] [--drop-match-only] [--write]
```

`CONFIG` is the reference, run from its own file as it is, and needs exactly one `nearest_player` source only for the
classic home variants (`home_visible`, `home_all`, `alive_home_visible`, `alive_home_all`), which replace it; `alive`,
`restart`, `restart_run` and the `plus_` variants need none. The runner's table then says, against it, what each variant gains:

| config | evidence |
|---|---|
| `<stem>__alive` | CONFIG's own, plus `oracle_alive`: a perfect dead-ball detector |
| `<stem>__home_visible` | `nearest_player` replaced by `oracle_home_visible`: the true possessor on the frames where the tracking has the ball, nothing where it does not |
| `<stem>__home_all` | `nearest_player` replaced by `oracle_home_all`: the true possessor on every live frame, the ball seen or not |
| `<stem>__alive_home_visible` | both of the above |
| `<stem>__alive_home_all` | both, whole: the truth itself, which a decoder free to follow it reproduces (exactly 1.0 on all three scores for an hmm with switch penalties near 0 and minimums of 1) |

The oracles are exact: `oracle_alive` is 1 on the truth's `H` and `A` and 0 on its `D`; the home oracles are 1 on `H`
and 0 on `A`, the visible one only where `atd.ball_present` (as the profile degraded it: on `tracking`, where A10 and
A13 were measured, the ball is missing where ours was), and every channel they do not speak on is NaN. A source that replaces `nearest_player` keeps its weight,
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
  `Config` holds) is refused by name before any file is written, and so is a config without one `nearest_player` when a
  classic home variant is asked for (`alive`, the restart and the `plus_` variants need none). What
  only the runner knows (an unknown profile, say) is refused by it, after the variants are written.
- **Reports** are written only with `--write`, and only for the variants run, as the runner writes them
  (`<stem>__<variant>[.<truth>][.m-<ids>][.s<n>].json` under `<data dir>/runs/benchmark/<profile>/`). `CONFIG`'s own
  report is `run.py`'s to write: it holds the comparison with whichever configs it was run beside, which a headroom run
  cannot reproduce, so a headroom run never writes or replaces it.

**`--variants` (P17).** `--variants NAME ...` picks the variants to run, by suffix, from the five above, two more:
`restart` and `restart_run`, and the six `plus_` ones below. Without it the five run, as always. The two restart variants replace nothing, so they need
no `nearest_player` (`main.toml` has none), and the `plus_` ones neither; only the classic home variants are refused for a config without one, by name. Each adds an
oracle to CONFIG's own evidence at weight 1, and both read the truth by closure like the rest:

- `oracle_restart` (`main__restart`) is a perfect "which team restarts" for 2 s. After each true `D` run followed by `H`
  or `A` in the same graded half, `home` is the restarting team (1 for `H`, 0 for `A`) from the first live frame for
  `RESTART_TAIL` = 50 frames, cut short where the truth next changes and at the block's end. A `D` run that opens a block
  (the kick-off) counts; one followed by `X`, or running to the block's end, restarts nothing. Everything else is NaN,
  `alive` included. It is the go/no-go number of P17 (A12).
- `oracle_restart_run` (`main__restart_run`) is the same held until the possession ends, the next frame whose code
  differs. Context only: it also gives the decoder where the first possession ends, which is not "who restarts", so it
  overstates what a restart source could do.

```
.venv/Scripts/python pipeline/benchmark/headroom.py pipeline/configs/main.toml --variants restart restart_run --truth raw --write
```

**The `plus_` variants (P18).** For a config with several home sources and no `nearest_player` (`main.toml` has
`controlled_touch` and `shape_possession`), the classic home variants have nothing single to replace. The `plus_`
variants replace nothing either: each is CONFIG's own evidence, then `oracle_alive` if the name starts with `alive_`,
then one home oracle at weight 1. `--variants` takes their suffixes:

| suffix | evidence added to CONFIG's own |
|---|---|
| `plus_home_visible` | `oracle_home_visible`: the true possessor where the tracking has the ball |
| `plus_home_gaps` | `oracle_home_gaps`: the true possessor only where the ball is missing (`~atd.ball_present`), NaN where it is seen; with the visible one it makes `oracle_home_all` |
| `plus_home_all` | `oracle_home_all`: the true possessor on every live frame |
| `alive_plus_home_visible`, `alive_plus_home_gaps`, `alive_plus_home_all` | the same, with `oracle_alive` added before the home oracle |

Adding is close to replacing where the oracle speaks: it carries about 13.8 nats a frame at the default `eps` against at
most about 2.6 for a real source, so the decoder follows it, and where it is silent the config's own sources decide.
`alive_plus_home_all` reproduces the truth only for a decoder free to follow it; on `main.toml` it scores 0.918 raw and
0.969 coarse (A13). A config that has a `nearest_player` keeps it under `plus_`, and the
classic variants still replace it; neither changes the other.

```
.venv/Scripts/python pipeline/benchmark/headroom.py pipeline/configs/main.toml --variants plus_home_visible plus_home_gaps plus_home_all --truth raw --write
```

From R017 to R029 `main.toml` had a match-only source (`smart_tagging`: DFL has no tags), so headroom on it needed
`--drop-match-only`, as `run.py` does; since R029 (3 Nov) the flag is a no-op on `main.toml`, and a config with the tags
(`r024_both.toml`) still needs it: `headroom.py pipeline/configs/main.toml --drop-match-only --variants plus_home_visible
plus_home_gaps plus_home_all`. The flag goes to the runner for CONFIG and every variant, and the output names what it dropped;
the headroom is then that of `main.toml` without the tags.

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
`test_headroom_writes_the_variants_reports_only_when_asked_and_never_the_references`,
`test_headroom_command_line_prints_the_table_and_the_report_lines`,
`test_headroom_command_line_drops_match_only_sources_when_asked`,
`test_headroom_gaps_oracle_speaks_only_where_the_ball_is_missing`,
`test_headroom_plus_variants_add_the_oracles_after_the_configs_own_sources`,
`test_headroom_plus_variants_add_and_classic_ones_still_replace`, `test_headroom_alive_plus_home_all_is_perfect` and,
for the restart variants, `test_headroom_restart_oracle_marks_each_restart`,
`test_headroom_restart_oracle_stops_at_the_next_change_and_the_block`,
`test_headroom_restart_oracle_needs_a_team_after_the_stoppage`,
`test_headroom_restart_variants_need_no_home_source` and `test_headroom_restart_variants_run_end_to_end`, all in
`checks/headroom.py` and all synthetic.

## Restarts: how the stoppages end

`restarts.py` (P17) says how big the thing a restart-team channel would fix is, and how much of it the pipeline already
gets. After a ball out of play, a foul or a goal the other team often restarts, and a decoder that carries the team
across a stoppage is then wrong at the start of the next possession. It has five modes and prints and writes aggregates
only (the first two are the anatomy of A12 part 1, the last three belong to the source `restart_team` and A12 part 2):

```
.venv/Scripts/python pipeline/benchmark/restarts.py anatomy CONFIG [--profile P] [--seeds N] [--truth raw|coarse] [--matches ID ...] [--data-dir D] [--out PATH]
.venv/Scripts/python pipeline/benchmark/restarts.py bar-ath CONFIG [--data-dir D] [--out PATH]
.venv/Scripts/python pipeline/benchmark/restarts.py stats [--model M] [--profile P] [--seeds N] [--matches ID ...] [--data-dir D] [--out PATH]
.venv/Scripts/python pipeline/benchmark/restarts.py events [--model M] [--tail T] [--use-ball] [--profile P] [--seeds N] [--matches ID ...] [--data-dir D] [--out PATH]
.venv/Scripts/python pipeline/benchmark/restarts.py configs STATS_JSON --out-dir DIR [--main pipeline/configs/main.toml] [--span after|stoppage_end] [--from-tail TAIL] [--switch-restart X ...]
```

- **`anatomy`** runs on the benchmark (the runner's default profile, `generic` since R030, and 3 seeds; A12's figures
  are `tracking`'s). The pipeline's labels are made as the runner
  makes them (the degraded copy, `MatchInputs` with no flag, `run.predict`), and the truth is read only to find the
  restarts, a restart being a maximal `D` run inside a graded block followed by `H` or `A`. For each match and pooled,
  by sums over matches and seeds (a restart is counted once per seed), it reports the number of restarts and the
  median stoppage in seconds; the **flip share**, the restarts whose team differs from the truth's team on the frame
  before the stoppage (those with no team before it left out), which is the part a carried team gets wrong by
  construction; and the **outcome** on the 50 frames after the restart, *right* (more frames of the restarting team than
  of the other), *wrong* (some `H`/`A` and not right; a tie is wrong) or *still D* (none), as shares of all restarts and
  split by flip and no flip. The coarse truth gave the same figures to the last digit on `main.toml` (A12).
- **`bar-ath`** is label-free and has no truth: it counts `CONFIG`'s own predicted stoppages on our match (a `D` run
  followed by `H` or `A`, per half) and for each whether the ball is missing on more than half of the 50 frames before it
  and after it. It says how much of the time a rule that reads the ball around a restart would have no ball, which is
  the share against which the `tracking` profile's ball gaps, a transplant of ours, could be held (A12). Only counts and shares leave it,
  never a frame or a label sequence.

- **`stats`** measures what a vote of each restart type is worth, on the benchmark (the source `restart_team` run on the
  degraded tracking of every match and seed, for each setting of a grid of `tail` {25, 50, 100, 150} x `use_ball`). Per
  type, pooled by sums over matches and seeds: the **agreement** (voted frames whose truth is `H` or `A` and equals the
  called team, over voted frames whose truth is `H` or `A`), the calls per seed, the **coverage** (true restarts with a
  call of the type within 50 frames) and the **confidence**: the agreement floored to a hundredth, or none (the type does
  not vote) if the agreement is below 0.60 or there are fewer than 30 calls per seed (the rule, declared in the design
  before any measurement). Nothing is tuned: the grid is read and the rule applied. The truth only does the counting.
- **`events`** is a diagnostic that chooses nothing. For one setting (`--tail`, `--use-ball`), each DFL restart event
  (kick-off, throw-in, corner, goal kick, free kick, penalty) is matched to the nearest call within 50 frames, and it
  reports the recall per kind and the confusion of the call's type against the event's kind. The events are DFL's own
  human-collected event data of the seven matches (below), and no constant comes from them.
- **`configs`** reads the JSON of `stats --out` and writes one config per setting, `r014_t{TTT}_b{0|1}.toml`: `main.toml`
  as it is plus a `restart_team` source with that setting's measured confidences (a type without one does not vote).
  These are R014's eight configs, in `pipeline/benchmark/configs/r014/`. With `--span stoppage_end --from-tail TAIL`
  the source has no tail and votes one observation spread over the stoppage that ends at the restart, at the
  confidences of the tail's `t{TAIL}_b0` and `_b1` settings; `--switch-restart X ...` (inf allowed) also writes the
  decoder's `restart_source` and `switch_restart`, one config per value: R014b's eight, `r014b_s{XX}_b{0|1}.toml`, in
  `pipeline/benchmark/configs/r014b/`.

Reports go to the paths of `--out`, by convention `data/runs/benchmark/restarts/anatomy.<truth>.json` and
`data/runs/restarts/bar_ath.json` (both under the gitignored `data/`). Its results are A12's (the first two modes, part 1; `stats` and `events`, part 2).

## P04's control rule: the measurement and Stage A's configs

`controlled_touch` (P04, `pipeline/evidence/controlled_touch.py`) speaks only inside control episodes, and its
`confidence` is a measurement of the benchmark, not a tuning (Stage A's protocol, as R008 describes it).
`touch_stats.py` makes it:

```
.venv/Scripts/python pipeline/benchmark/touch_stats.py [--grid stage_a|r013] [--profile P] [--seeds N] [--matches ID ...]
                                                       [--data-dir D] [--out PATH]
result = touch_stats.measure(settings, profile=..., seeds=..., match_ids=None, matches=None, calibration=None)
```

For each of the 12 settings (radius 1.0, 1.5, 2.0 m x `max_rel_speed` 3.0, 6.0 m/s x `min_frames` 3, 8: `DEFAULT_GRID`,
`--grid stage_a`, the default), or of R013's 48 (radius 1.5 to 3.0 m x 6, 9, 12 m/s x 1, 2, 3, 5 frames: `R013_GRID`,
`--grid r013`), it runs the source once per match and seed on the degraded tracking (the runner's default profile,
`generic` since R030; seeds 0 to 2), and counts, among the alive graded frames where the served tracking has the ball:

- **coverage**, the share inside a kept episode, and **agreement**, the share of those whose team is the truth's, per
  match and pooled over matches and seeds;
- the **flip lags**: for each change of team in the truth with no `D` between, at frame c, the frames from c to the
  start of the new team's kept episode that begins nearest to c (the earlier on a tie), among the starts in
  [max(c - 5 s, p), min(c + 5 s, n)] of the same half, where p and n are the previous and the next change of the truth
  code of any kind (`D` included), or the half's ends. The window never reaches past a neighbouring change, so a flip is
  never matched to an episode of another possession; no start in it means not found. Median, 25th and 75th percentile
  and the share of changes with one found. A negative lag means the episode began before the flag turned, so a flag
  that turns late shows as a negative lag and an early one as a positive lag. It is a diagnostic of DFL's convention,
  read beside the `hold` result, never a selector;
- the **confidence**: floor(100 x pooled agreement) / 100 clipped to [0.51, 0.99] (`confidence_from_agreement`). It is
  a measurement on all seven matches, and the entry that uses it says so.

The truth is read inside `measure`, from the match, to count; the source is called on a `MatchInputs` that holds the
degraded tracking and no flag, with confidence 0.99 and no hold, and the result holds counts and shares only. The CLI
prints a table of aggregates and nothing frame-level, and `--out` also writes the result as JSON (R013's configs read
their confidences from it).

Stage A's configs are named `pipeline/benchmark/configs/p45/p45a_r{RR}_v{V}_k{KK}_h{HH}.toml`: `RR` the radius
in tenths of a metre (`10`, `15`, `20`), `V` the `max_rel_speed` in m/s (`3`, `6`), `KK` the `min_frames` (`03`, `08`),
`HH` the hold in hundredths (`50` is off, i.e. 0.5, and `55` is 0.55); `RR`, `KK` and `HH` are zero-padded to two digits
and `V` is one digit, e.g. `p45a_r10_v3_k03_h50.toml`. The 24 files each hold `controlled_touch` with the measured
confidence, `ball_presence` at its defaults and `hmm` at (8, 32, 75, 1). The controls are `p45actl_nearest` and the
chosen config on `context`. The checks are in `checks/touch.py` (synthetic): `test_touch_stats_matches_a_loop`,
`test_touch_stats_confidence_rule`, `test_touch_stats_default_grid`, `test_touch_stats_r013_grid`,
`test_touch_stats_never_passes_truth` and `test_touch_stats_refuses_what_it_cannot_measure`.

Stage B's configs (R009) are `p45b_{src}_w{WWW}_bp{B}.toml`: `src` is `hgb`, `logit` or `rules`, `WWW` the weight in
thousandths (`010` to `100`) and `B` is 1 with `ball_presence` at its defaults, 0 without; each holds `nearest_player`
at 0.78 and `hmm` at (8, 32, 75, 1) (24 files).

Stage C's configs (R010) are `p45c_t{TTT}_d{DDD}_mp{MMM}_md{NNN}.toml`: the chosen `controlled_touch` and
`dead_ball_model` alone, with `hmm` at `switch_team` TTT, `switch_dead` DDD, `min_possession` MMM and `min_dead` NNN
(72 files).

P07's grid (R011) is `configs/p07/p07_w{WWW}_{gaps,all}.toml`: `WWW` is the weight of `shape_possession` in
thousandths (`010`, `020`, `050`, `100`, `200`, `500`) and `gaps` or `all` its scope; each is R010's `main.toml` (now
`pipeline/configs/p45_hmm.toml`: `controlled_touch`, `dead_ball_model` and `hmm` (4, 32, 25, 25)) plus the source with
`pipeline/models/shape_possession_hgb.json` and `min_prob` 0.02 (12 files). The chosen one, `p07_w200_all`, is
`pipeline/configs/p07_shape.toml`. P02's guard (R012) is `configs/p02/p02_sb.toml`: R011's config plus `shift_edges`
at the pipeline's own lag on DFL (team 1, into_dead -2, out_of_dead 0), run against `p07_shape.toml` in four cells (the
section on [conventions](#conventions-when-our-truth-flips-and-the-edge-shift) has the rule).

R013's grids are in `configs/r013/`, each `main.toml` as of R012 with one part changed. Stage A,
`r013a_r{RR}_v{VV}_k{K}_h{HH}.toml`, varies `controlled_touch`: `RR` the radius in tenths of a metre (`15` to `30`),
`VV` the `max_rel_speed` in m/s (`06`, `09`, `12`), `K` the `min_frames` (`1`, `2`, `3`, `5`) and `HH` the hold in
hundredths (`52`, `55`, `60`, `70`), each at the confidence `touch_stats.py --grid r013` measured (192 files;
`r013a_r20_v06_k3_h55` is `main.toml`'s own setting). Stage C, `r013c_t{TT}_mp{MMM}_w{WWW}.toml`, keeps Stage A's
choice and varies the `hmm`'s `switch_team` `TT` and `min_possession` `MMM` and `shape_possession`'s weight `WWW` in
thousandths (144 files).

R030's configs (P27) are in `configs/p27/`, written by `p27_grid.py` and never by hand; the section on
[P27's re-choice](#p27s-re-choice-on-generic-r030) has the tools. They are R013's two grids again, on `generic`:
- `p27_ref`: the incumbent constants (R029's `main.toml` without the shift), the reference of Stages A and C;
- Stage A, `p27a_r{RR}_v{VV}_k{K}_h{HH}.toml` (192 files, named as R013's), each at the confidence that
  `touch_stats.py --grid r013 --profile generic --seeds 3` measured;
- Stage C, `p27c_t{TT}_mp{MMM}_w{WWW}.toml` (144 files) at Stage A's choice;
- Stage D, `p27d_base` (Stage C's choice, the incumbent of every group) beside `p27d_sd{SS}_md{MM}`,
  `p27d_dw{WWW}` (the dead-ball weight in hundredths), `p27d_scope_gaps` and `p27d_noshape`;
- `p27_preshift` (the chosen config before the shift) and `p27_final` (with it), whose constants `main.toml` carries
  since R030.

## Choosing from a grid

```
.venv/Scripts/python pipeline/benchmark/choose.py --reference main STEM [STEM ...] [--profile P] [--band 0.005]
                                                  [--metric spell|macro] [--objective raw|mean] [--top N] [--data-dir D]
                                                  [--out PATH]
choice = choose.choose(stems, reference="main")
```

`choose.py` applies the selection rule that R008 to R013 declare to the reports `run.py` wrote (both truths, so run the
grid with `--truth both`): the best pooled raw spellF1, every config within the tie band (0.005) of it tied, the best
pooled coarse spellF1 among the tied, and a tie that is left to the stem named first, so the stems are given in the
declared tie order. It prints the table (every config, or the `--top` best by raw score) with each config's pooled Δ,
sign-flip p and wins against the reference under both truths, the choice and its Δs, and the leave-one-match-out
estimate that R008 to R011 worked out by hand: for each match, the selection repeated on the other six (the raw
objective alone, and the whole rule on six-match means) and the chosen config's score on the held-out match against
the reference's. Every number is computed from the reports' halves with `score.py`, so it is the runner's. A report is
refused if its config file has changed since it was written, and reports of different runs (matches, seeds, profile or
Calibration) are refused together.

`--metric macro` (`metric="macro"`, P12) judges the grid on pooled macroF1 in place of spellF1, for a refiner that can
only move macroF1 (`free_x`, since removed, R016): the raw macroF1 selects, the coarse macroF1 breaks a tie, the Δs, p and wins in the
table and the `versus` lines are macroF1's, the lomo averages each match's macroF1, and the table's header names the
metric. The band is in the metric's units, so such a grid declares it (R016 passes `--band 0.002`); the default stays
0.005, and every other field is unchanged. The checks are `test_choose_applies_the_rule_and_matches_the_runner`,
`test_choose_selects_on_macro_when_asked` and `test_choose_refuses_missing_stale_and_mixed_reports`
(`checks/choose.py`, synthetic).

`--objective mean` (`objective="mean"`, R024) is the rule of every choice declared after R023, whose outcome was
"both". The objective is the mean of the raw and coarse pooled scores, and every config within the band of the best
is tied. Among the tied, the larger min(raw, coarse) wins: the config that does best under whichever truth is worse for
it. A tie that is left goes to the stem named first. The rows, the `versus` lines and the lomo gain a mean entry: its Δ
is the mean of the two truths' Δs, and its p and wins are taken on the per-match means. The lomo's whole rule uses
the six-match min(raw mean, coarse mean). `--objective raw`, the default, is R008 to R016's rule to the bit (a check
compares it with the code before the change).

`--profile` defaults to the runner's, so since R030 `choose.py` reads the reports under `runs/benchmark/generic/`.
The grids of R008 to R024 were run on `tracking`: their reports are read with `--profile tracking`, as a record and
not for a new choice.

## P27's re-choice on `generic` (R030)

Four tools re-run the whole re-choice from the repository. Every run is `--profile generic --seeds 3`, the seven
matches, both truths, and the objective is the mean of the two truths' pooled spellF1. The profile and every rule
were declared before the runs ([the spec](../../docs/superpowers/specs/2026-11-09-p27-degradation-design.md)); what
they chose is [R030](../../docs/results.md).

```
.venv/Scripts/python pipeline/benchmark/p27_grid.py ref   --confidences FILE
.venv/Scripts/python pipeline/benchmark/p27_grid.py a     --confidences FILE
.venv/Scripts/python pipeline/benchmark/p27_grid.py c     --confidences FILE --touch R,V,K,H
.venv/Scripts/python pipeline/benchmark/p27_grid.py d     --confidences FILE --touch R,V,K,H --decoder T,MP --weight W
.venv/Scripts/python pipeline/benchmark/p27_grid.py final --confidences FILE --touch R,V,K,H --decoder T,MP --weight W \
    [--switch-dead SD] [--min-dead MD] [--dead-weight DW] [--scope S] [--no-shape] [--shift TEAM,INTO_DEAD,OUT_OF_DEAD] [--name STEM]
.venv/Scripts/python pipeline/benchmark/p27_choose.py a|c|d [--data-dir D]
.venv/Scripts/python pipeline/benchmark/p27_choose.py bar STEM
.venv/Scripts/python pipeline/benchmark/p27_choose.py guard STEM BASE
.venv/Scripts/python pipeline/benchmark/p27_shift.py STEM
.venv/Scripts/python pipeline/benchmark/p27_profile_check.py [--seeds 3] [--data-dir D]
```

- **`p27_grid.py`** writes the configs of a stage into `configs/p27/` and prints their stems in the stage's tie
  order. `FILE` is what `touch_stats.py --grid r013 --profile generic --seeds 3 --out FILE` wrote: a touch setting's
  confidence is read from it and never typed in. It reads no match. `--prefix P` (stages `c` and `d`, here and on
  `p27_choose.py`; default `p27`, the declared branch) names the stems of a labelled alternative branch beside the
  declared one.
- **`p27_choose.py a`, `c` and `d`** apply the declared rule to the stage's reports through `choose.py`'s mean
  objective: the best mean, a tie band of 0.005, the larger min(raw, coarse), then the stage's tie order, against the
  reference `p27_ref`. Stage D takes each group by the same rule, Stage C's choice being one candidate, and the shape
  source stays if the config with it is not worse than the config without it. Each prints aggregates of the DFL
  benchmark and a last line that starts with `NEXT`, holding the arguments of the next `p27_grid.py` call, and writes
  its selection as JSON under `<data dir>/runs/p27/`.
- **`p27_choose.py bar STEM`** is R010's bar: `STEM` must beat `nearest_w50` under the raw truth and `nearest_w125`
  under the coarse truth, each with a sign-flip p of at most 0.05 over the seven matches. It exits 2 if the bar fails.
- **`p27_choose.py guard STEM BASE`** is the shift's guard: the shifted config is kept only if its mean is above the
  unshifted one's and neither truth falls by more than 0.005.
- **`p27_shift.py STEM`** reads the latency file that `conventions.py latency pipeline/benchmark/configs/p27/STEM.toml
  dfl --profile generic --seeds 3 --truth raw` wrote, and prints, per type of edge, the changes, the matched ones, the
  median lag and the shift by R012's rule: minus the rounded median, and 0 below 30 matched changes. It reads DFL's
  lags and nothing of our match. `conventions.py shift` also reads the public window, which is why R030 did not use
  it.
- **`p27_profile_check.py`** degrades each match under each seed again and sums the counts the degrader records (no
  frame is kept or printed). It passes if the pooled mask share is 0.41 ± 0.02 and the pooled share of on-pitch
  player-frames lost is 0.068 ± 0.01, and exits 2 otherwise.

The order R030 ran them in: both models retrained (`train_dead_ball.py`, `train_shape.py`), the touch rule's
confidences measured (`touch_stats.py`), then `ref` and Stage A, Stage C at Stage A's choice, Stage D at Stage C's,
the shift from the pre-shift config's lag, the guard, and the bar on the final config. One heavy process runs at a
time.

## A19's diagnosis: the ladder and the calibration

```
.venv/Scripts/python pipeline/benchmark/diagnose.py CONFIG [--profile P] [--seeds N] [--matches ID ...] [--drop-match-only]
                                                    [--data-dir D] [--no-write]
```

`diagnose.py` (A19, P23 B and D) runs a config on the benchmark once and writes `<data dir>/runs/diagnose/<stem>.json`,
aggregates of DFL only. It has two parts.
- **The ladder.** For each truth it decomposes where spellF1 goes between IoU 0.3 and 0.7: every pair the scorer
  matches at 0.3 is clean, merge, split or both by its same-team overlaps, and each class's loss up the ladder is
  given in spellF1 units. The clean pairs' edge errors are reported too.
- **The calibration.** Each source and the hmm's posterior against the raw truth: log-loss, Brier, ECE, the reliability
  table and a Platt fit (α below 1 is overconfident).

Its matching is the scorer's own (`pm.matched_gt_flags`), and the code checks it on every half. Its pooled spellF1
must equal `score.pooled`'s.

`--profile` defaults to the runner's, and the report of the default profile is named `<stem>.json` with no profile in
it. So a bare `<stem>.json` written before 10 Nov 2026 is A19's `tracking` report, and one written since R030 is
`generic`'s.

## Dead-ball model

P05's classifier is trained here, on the benchmark's matches, and run by the pipeline from a file (`pipeline/models/`,
JSON, no pickle). The split is the point: training needs scikit-learn and the seven DFL matches and lives in
`pipeline/benchmark/train_dead_ball.py`; `pipeline/evidence/dead_ball_model.py` reads the numbers back and runs the
trees with numpy alone, so the submission pipeline's pins stay numpy and pandas.

```
.venv/Scripts/python pipeline/benchmark/train_dead_ball.py --model hgb|logistic \
    [--profile generic] [--seeds 3] [--stride 5] [--out PATH] [--verify] [--data-dir D]
.venv/Scripts/python pipeline/benchmark/train_dead_ball.py --rules [--profile generic] [--seeds 3] [--data-dir D]
```

- **Rows.** Every match under every seed of the profile (default `generic`, version 1, since R030; `tracking`,
  version 3, until then; seeds 0 to 2) is served as
  `MatchInputs`, and `play_features` computes the 30 features from those inputs, so no truth reaches them
  (`test_trainer_never_passes_truth`). A row is every `stride`-th graded frame of a block from its start (default 5,
  which is deterministic and cuts the autocorrelation), labelled 1 where `truth.truth_codes` is not `D`.
- **Models.** `hgb` is `HistGradientBoostingClassifier` with fixed settings that are never tuned: log-loss, learning
  rate 0.05, 300 rounds, 15 leaves, at least 1,000 rows a leaf, L2 1.0, 255 bins, `random_state=0`, no class weights,
  no early stopping (its random hold-out of autocorrelated frames leaks), monotone constraints of -1 on
  `ball_outside_m` and `slow_share_c4`. `logistic` is the reference: a missing indicator and mean imputation for each
  feature with a NaN, constant columns dropped (and recorded), then `hazard.fit_logistic(l2=1.0)`.
- **Folds.** For each match, a model fitted on the other six (`folds`), and `full` on all seven. The source scores a
  benchmark match with the model that never saw it and BAR-ATH with `full`;
  `test_trainer_folds_exclude_the_held_out_match` pins the bookkeeping.
- **The file.** `{format, kind, feature_version, features, profile, seeds, stride, provenance, folds, full}`, sorted
  keys, Python's float text (reads back exactly), LF, no NaN, under 10 MB, no frame-level data. An `hgb` model is its
  baseline and, per tree, the arrays `feature`, `threshold`, `missing_left`, `left`, `right`, `leaf` and `value`, read
  out of scikit-learn's fitted predictors (a private structure, hence the exact pins in `requirements-benchmark.txt`);
  `test_exported_trees_match_sklearn` holds the numpy walk of the exported trees to 1e-12 of `predict_proba`, NaNs
  included, and `test_training_is_deterministic_across_threads` that one thread and four write the same bytes.
- **Provenance** is the scikit-learn version, the Calibration's digest (null for a model trained on `generic`, which
  takes no Calibration), the DFL files' sha256 and the sha256 of the
  three source files the model depends on (`train_dead_ball.py`, `play_features.py`, `kinematics.py`), and no git
  commit, which would change with every later commit. **`--verify`** retrains and compares with the committed file
  everything but those three hashes: a different hash is a note (a comment edited after training is not a different
  model), anything else exits 1 (`test_verify_ignores_only_source_hashes`).
  The comparison is exact, and the features' BLAS products can differ in the last bit on another CPU family, so
  `--verify` is a same-machine check.
- **What it prints**, per held-out match on its out-of-fold predictions: log-loss, Brier score and AUC, and a pooled
  10-bin reliability table, counts and means only.
- **`--rules`** measures `dead_ball_rules`' five parameters: for each trigger and for "no trigger", on all graded
  frames, P(trigger | alive) and P(trigger | dead), and the value P(t | alive) / (P(t | alive) + P(t | dead)) rounded
  towards 0.5 to two decimals (`test_rule_values_rule`). The triggers are read off the rules by probing them, never
  written twice (`test_rule_counts_use_the_rules_own_triggers`). They are measured on all seven matches, which is a
  small optimism.

The source takes the file's model for `inputs.game` or else `full`, and gives the decoder a likelihood, not a
posterior: `logit(alive) = logit(p) - logit(prior_alive)`, clipped to `[min_prob, 1 - min_prob]`; the decoder sets the
prior, and DFL's class balance does not leak into it.

**The committed artefacts.** `pipeline/models/dead_ball_hgb.json`, the one `main.toml` reads, is R030's: retrained on
`generic` with the settings above (pooled held-out log-loss 0.0788 and Brier 0.0227 over 601,602 rows, AUC 0.9942 to
0.9973 per match). R009's version of it, trained on `tracking` v3, is in the git history, and its scores are R009's.
`pipeline/models/dead_ball_logistic.json` is still R009's reference, trained on `tracking` v3, and no config of the
submission reads it. R030 retrained the logistic reference on `generic` into a scratch file, for its comparison only
(pooled held-out log-loss 0.1524; the trees are lower in 7 of 7 matches, so they stay), and did not commit it.

## Shape model

P07's classifier (which team has the ball, from the shape of the two teams) is trained here the same way, and run by the
pipeline from a file: training needs scikit-learn and the seven DFL matches and lives in
`pipeline/benchmark/train_shape.py`; `pipeline/evidence/shape_possession.py` reads the numbers back and walks the trees
with numpy alone (`pipeline/evidence/trees.py` is the walk, shared with `dead_ball_model`).

```
.venv/Scripts/python pipeline/benchmark/train_shape.py [--profile generic] [--seeds 3] [--stride 5] [--out PATH] \
    [--verify] [--report] [--data-dir D]
```

- **Rows.** Every match under every seed of the profile (default `generic`, version 1, since R030; `tracking`,
  version 3, until then; seeds 0 to 2) is served as
  `MatchInputs`, and `shape_features` computes the 26 features from those inputs, so no truth reaches them: the label
  is read from `truth.truth_codes` beside them. A row is every `stride`-th frame (default 5) of a graded block from its
  start, kept where the truth is `H` or `A` (who has the ball is not asked of a dead ball), and labelled 1 where it is
  `H`.
- **The mirror.** Every row is also added as `shape_features.swap(row)`, the same frame with the two teams exchanged,
  with the label flipped. The classes then balance exactly (`prior_home` is 0.5, which the source requires) and the
  model is taught the symmetry; the source makes its output exactly antisymmetric, `(r(x) - r(swap(x))) / 2`, so there
  is no prior to subtract.
- **The model.** `HistGradientBoostingClassifier` with P05's settings, imported from `train_dead_ball.HGB_SETTINGS`
  and never tuned (T26): log-loss, learning rate 0.05, 300 rounds, 15 leaves, at least 1,000 rows a leaf, L2 1.0, 255
  bins, `random_state` 0, no class weights and no early stopping. Unlike P05's, no monotone constraint: nothing in the
  laws of the game says that the team with the ball must be further up the pitch or wider.
- **Folds and the file.** For each match a model fitted on the other six (`folds`, keyed by the DFL match id, which is
  `inputs.game` of a served match), and `full` on all seven. A benchmark match is scored by its own fold and BAR-ATH by
  `full`. The file is `pipeline/models/shape_possession_hgb.json`, in P05's layout (`{format, kind, feature_version,
  features, profile, seeds, stride, provenance, folds, full}`; sorted keys, LF, no NaN, under 10 MB), 3,668,856 bytes
  in R011's version. The committed file is R030's, retrained on `generic` with the same settings. The trees are
  exported by `train_dead_ball.export_hgb`, so the version of scikit-learn is pinned for it too.
- **Provenance and `--verify`.** The provenance is the scikit-learn version, the Calibration's digest (null for
  `generic`), the DFL files'
  sha256 and the sha256 of the three source files the model depends on (`shape_features.py`, `trees.py` and the
  trainer), and no git commit. `--verify` retrains and compares everything but those three hashes with the committed
  file; a different hash is a note, anything else exits 1. It is a same-machine check, as P05's is.
- **`--report`** scores each fold model on its own held-out match, served as the pipeline serves it (`shape_possession`
  with scope `all`), seeds pooled per match and the matches pooled by frames: log-loss and accuracy on the alive frames
  and on those whose served ball is missing, `ln 2`, the frames where the source is silent (counted apart), and
  `touch_agreement`. It says whether the stop rule P07's design declared fires: the pooled held-out log-loss on the
  alive frames with the ball missing must be below `ln 2 - 0.01` = 0.6831. A fold missing from the artefact is an error,
  because the source would fall back to `full`, which saw the match. R011's part 1 has the numbers on `tracking`.
  On `generic` (R030) the pooled held-out log-loss on the alive frames with the ball missing is 0.3560 (accuracy
  0.8458 over 685,247 frames), so the rule does not fire.
- **`touch_agreement(inputs, model)`** is the statistic that needs no truth: among the frames inside kept
  `controlled_touch` episodes (`conventions.TOUCH`, confidence 0.99, hold off, so that `home` is a number exactly on
  the kept frames), the share where the shape `home` (scope `all`, at least 0.5) names the episode's team. It takes any
  `Inputs`, so it can be asked of our own match with `full`: R011's transfer gate (0.8251 on BAR-ATH against 0.7486,
  the lowest held-out match, with the model trained on `tracking`). For R030's model the lowest held-out match is
  0.7426 (pooled 0.8334), so the gate on our match is at least 0.7426; its result is in R030. It is a check that
  passes or fails, a proxy, and not an accuracy.

The checks are in `checks/shape.py` (synthetic): the rows, the augmentation, the exported trees against scikit-learn's
raw score, the same bytes on one and four threads, the folds, that no truth reaches the features, the artefact, the
report, and `touch_agreement` on a hand-built scene.

## Conventions: when our truth flips, and the edge shift

P02's instrument, `pipeline/benchmark/conventions.py`, measures when the label of a truth changes (A11) and when the
pipeline's own label does (R012). No LLM may see the match, so the timing is measured against events that the tracking
shows without any label. Since the organisers' rule of 23 Oct 2026 (`f8f2629`) the public window is inspected with it,
to understand how our truth flips, and never tuned on: it is "not training data and not a tuning set".

```
.venv/Scripts/python pipeline/benchmark/conventions.py public  [--gt PATH] [--data-dir D]
.venv/Scripts/python pipeline/benchmark/conventions.py dfl     [--profile generic] [--seeds 3] [--truth raw|coarse] \
    [--matches ID ...] [--data-dir D]
.venv/Scripts/python pipeline/benchmark/conventions.py compare [--profile generic] [--truth raw] [--data-dir D]
.venv/Scripts/python pipeline/benchmark/conventions.py latency CONFIG public|dfl [the options of public or dfl]
.venv/Scripts/python pipeline/benchmark/conventions.py shift --config CONFIG [--s-b-rejected TYPE ...] [--gt PATH]
```

- **Changes and events.** A change of a truth is typed `team` (H to A or A to H), `into_dead` or `out_of_dead`; a change
  into or out of `X` inside a block is refused. Eight detectors, label-free and each for the types it applies to:
  `touch_start` (the first frame of a kept `controlled_touch` episode of the new team), `line_out` (the ball first seen
  beyond a line), `ball_lost` (the first missing frame of a gap in the ball), `ball_moves` (the ball starting to move
  after 25 still frames, from positions), `model_dead` and `model_alive` (where `dead_ball_model`'s alive crosses 0.5)
  and, on BAR-ATH only, `tag_start` and `tag_end` (a Smart Tagging `SET PIECES` interval's ends). The lag is `event -
  change` in frames, so a negative lag means the event came first; the event taken is the nearest within 5 s, bounded by
  the neighbouring changes, and none in that range is "not found".
- **`public`** times the 20 changes of the public truth on our real tracking. It writes `public_lags.json` (read by
  `compare`) and `public_transitions.csv` under `data/runs/conventions/`: one row per change, with its frame and every
  detector's lag, for a person who may watch the video at exactly those moments, to understand them and not to fit a
  constant. Since 3 Nov 2026 ([docs/rules.md](../../docs/rules.md)) the person watches only to understand what counts
  as possession, at the truth's changes, and does not compare the detectors' events, which are our own output, with
  the video. No LLM reads that file and the code prints no row of it. **`dfl`** runs the same code on each DFL match as
  the pipeline sees it under a profile, with the truth read beside the inputs only to find the changes; DFL is pooled
  over matches and seeds by the lags themselves. **`compare`** prints the public median beside DFL's, with the gap's 90%
  bootstrap interval (2,000 draws of the public lags, seeded per cell, DFL's median taken as it is). A11 has the
  figures.
- **The descriptors** (A15) sit beside the lags: at each change frame the ball seen on it and within 1 s, its speed
  from the positions, its distance inside the line, and for a team change the nearest old-team and new-team slot and
  whether the new one is nearer. They go into both documents (`descriptors`, summarised by type; `descriptor_values`,
  the values `compare` resamples), into `compare` as `descriptor_gaps` (public minus DFL: a share for the yes/no ones,
  a median for the numbers, with the same 90% bootstrap), and into the person's CSV after the detectors' columns. They
  are for inspection only, they set nothing, and a printout of BAR-ATH hides quartiles and bounds below `MIN_SHOWN`
  values as it does for the lags.
- **What a printout of BAR-ATH may show.** A cell with fewer than `MIN_SHOWN` = 5 changes found prints n, the share
  found and the median only, and of an interval only whether it excludes 0, since the quartiles of two to four values
  fix every one of them. DFL's printouts are in full, and the JSON files keep every value.
- **`latency CONFIG`** turns the instrument on the pipeline's own output: a predicted change matches a true one of the
  same pair of labels within 50 frames, and the lag is predicted minus true, so positive means our label changed late.
  `public` reads the CSV that `pipeline/run.py CONFIG` wrote against the public GT; `dfl` runs the config on every
  benchmark match and seed. The config must decode per half. Each file records a sha256 of the config's content (scope,
  seed, evidence, decoder and refine, no comment), and the public one also of the CSV it was read from; `shift`
  refuses a file of another content or of none, with the `latency` command to run again.
- **`shift --config CONFIG`** is P02's declared rule, and nothing in it is tuned. For each type, s_b is minus the
  rounded median of DFL's latencies (the benchmark's own default profile and the raw truth: `conventions.SHIFT_DFL`,
  `("generic", "raw")` since R030 and `tracking` for R012 and R013; the coarse truth's
  is reported beside it, T33), and 0 below 30 matched changes. A guard on the benchmark confirms or rejects s_b as a
  whole (`--s-b-rejected` names the types whose guard failed). R012's guard was `configs/p02/p02_sb.toml` against
  R011's config on `tracking` and `context`, raw and coarse, kept only if the `tracking` raw gain is above 0 and no
  cell is below -0.005. R030's guard is the spec's, on `generic` alone (`p27_choose.py guard`): the shifted config's
  mean of the two truths above the unshifted one's, and neither truth more than 0.005 below. The rule R012 declared
  had a second term, g: minus the rounded median of the public latencies after s_b,
  taken only where that median is at least 3 frames in absolute value, its 90% bootstrap interval excludes 0 and every
  leave-one-out median has its sign; a single public change never sets a shift. **g is retired since the organisers'
  rule of 23 Oct 2026 (`f8f2629`)**: reading it off the public window is a tuning use of a window that is "not a tuning
  set", so it must stay 0. The code is unchanged and still computes and prints g, for reference only, and the shift to
  adopt is s_b alone, from the DFL matches (other games); any future shift comes from other games only (P16).
  `shift_edges` applies the shift. The public score of the shifted prediction is read in sample and out of fold (a
  boundary matched to change i moves by the shift learned without i), as a correctness check and never as a choice.

R012 applied the rule: s_b is team +1, into_dead -2, out_of_dead 0, g is 0 for every type (and is now retired), and
`main.toml` carries it. R030 derived s_b again on `generic` from the pre-shift config's lag (`p27_shift.py`, which
reads DFL's lags only and not the public window): median lags of -1, +2 and 0 frames over 2,967, 1,114 and 1,645
matched changes, so the same +1, -2 and 0. The guard kept it by a hair (mean +0.000029), and on spellF1 it is worth
nothing either way.
The checks are in `checks/conventions.py` (synthetic): the window rule against a plain loop, the detectors on hand-built
tracking, that the DFL mode reads the truth beside its inputs, that no transition frame is printed, the bootstrap, the
typed matching at the tolerance, each condition of the rule on a list that fails it, the out-of-fold prediction
against `shift_edges.apply`, and that a latency file of another content of the config, or of none, is refused.

## The checks

```
.venv/Scripts/python pipeline/benchmark/selftest.py          # synthetic: a minute or two
.venv/Scripts/python pipeline/benchmark/selftest.py --dfl    # + the seven downloaded matches (minutes)
.venv/Scripts/python pipeline/benchmark/selftest.py --data   # + our own match (aggregates only; minutes)
```

The synthetic run is 377 checks (P06's added 16 to the 272 of A05, then P04's touch_stats 5 and P05's trainer 19,
which made 312; then P07's trainer and source 20 in `shape`, P02's instrument and rule 40 in `conventions`, and one in
`run` that a config with `shift_edges` is scored like any other, 373; then R013's four: both truths from one decode,
R013's touch grid and the two of `choose`; that is the count as of R013, and the later modules, `generic`'s among
them, have added to it: R030 has the count of its own run). Its time, 49 to 115 s, is as of A05 and not measured
again. `--dfl` and `--data` are as of A05 (2026-10-23, `fdc5eae`), when the synthetic run was 272 checks: `--dfl`
added 16 and `--data` 7, and all of them (295 checks then) took 949 s (15 min 49 s) with the seven DFL caches and the
Calibration rebuilt after an edit to `dfl.py` and `calibration.py`. The machine's load decides, and the first run on a
machine also builds the caches (A05).

The checks live in `pipeline/benchmark/checks/`, one module per topic, listed in `checks.MODULES` (the folder's
`__init__.py` says what each holds): `sources`, `dfl`, `truth`, `inputs`, `score`, `run`, `choose` (choose.py's), `hazard`,
`calibration`, `degrade`, `cap`, `tracking`, `context`, `generic` (P27's profile), `fidelity`, `headroom` (the Headroom section's), `touch` (touch_stats's),
`dead_ball` (the Dead-ball model section's), `conventions` (the Conventions section's), `shape` (the Shape model
section's) and `harness` (the harness itself). The profiles' are `degrade` (the transplant worked out exactly,
determinism, what the `gaps` profile touches and what it does not, no truth, the order of the matches), `tracking`
(the tracklets, the episodes, the NaN-speed runs and the smoothing worked out exactly, and how it nests in `gaps`),
`cap` (the low-count tail's cap, its shortest-piece rule and the scale of the tracked spans), `context` (what the
hazard is given, the chance, the class, third and durations read at the at-risk frame, the counts of its record on
hand-built halves, the odds rescale and the frozen scale), `generic` (its constants are the declared ones; no
Calibration is taken or loaded by the degrader, the runner, the trainers or the instruments; no truth reaches it; the
ball's mask is the documented draws, laid from each kick-off; the players are lost at the declared rate in 46 slots;
the runner reports it with no Calibration; and `fidelity.py` refuses it) and `fidelity` (every gate and check worked out by other
code, pooling by sums, the exit code, the pending gates, the full-strength frames of the count gates and the gate
kinds). The one that anchors the rest is `test_dfl_oracle_scores_perfectly`: an evidence source that is the truth
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

Since R030 the default profile is `generic`, and its own limits are the first item. The items that name `tracking`,
`context`, the transplant, the cap or the tracklets describe the profiles that read our match: they are the limits
of R002 to R029's numbers, kept as that record.

- **`generic` is declared, not measured, so how far it looks like our match is unknown by design** (R030, the spec).
  - Its gaps ignore the play, and ours do not (DATA.md: the ball "goes missing disproportionately on the frames that
    matter"). So the benchmark still overstates the dead-ball model (T38, T39) and cannot value a feature about where
    the ball is lost.
  - It has no wrong-team slots and no long NaN-speed runs, which our tracking has (T31, T32). The old benchmark put
    the whole player degradation at 0.013 spellF1 (A14); the cost is not measured on our match.
  - The gap law is one declared range. Another range would give other constants.
  - It has no fidelity gate and is never run through `fidelity.py`. No profile may be changed to pass a measurement
    of our match ([docs/rules.md](../../docs/rules.md)).
  - The raw and the coarse truth still pull the hold and the decoder in opposite directions (T33), and both of R030's
    knife-edge choices sit on that trade-off.

- **Conventions and grain.** DFL's flags are one provider's, and ours may flip at other moments and cut possessions at
  another grain. DFL's flag is fine-grained, 437 possessions a match with a median of 3.7 s, and its `D` is its own
  ball-in-play clock to within 2 frames (A06). How fine our ground truth is cannot be told from the public window: the
  window is alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows, and against DFL's windows that lively
  it is 1.34 times slower, which DFL's own conventions can produce (A06, T33). `--truth coarse` stands in for a coarser
  grain and does not correct a measured difference, so a choice that depends on possession length (a smoothing level, a
  switch penalty) is ranked under both truths and a disagreement is a warning. `D` timing against ours is measured on
  four stoppages and read as a lead and not a measurement (A11, T40: our `D` looks 2.5 to 3 s longer than DFL's flag
  would draw it, from our truth's convention, chance or our tracker's coverage). The pipeline's own lag on DFL is
  shifted out (R012, 1 to 2 frames, inside noise), and a gap to our convention read from the public window came out 0
  (P02).
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
  or the tagging cannot be benchmarked here. It is checked on the public window for correctness only, with its strength
  fixed by a declared constant. Until 5 Nov the other way was to simulate it from the benchmark's truth with noise
  measured on our match (A03; R019 and R028 tried it and failed their gates). By T56's test a stand-in whose noise is
  measured on our match is fitting too, as the old profiles are, so it is not an option now
  ([docs/rules.md](../../docs/rules.md)).
- **Aerial projection.** A camera-based tracker projects an airborne ball onto the pitch plane, which displaces it.
  DFL's positions carry no such error and the benchmark adds none, and our ATD has no ball height, so the error cannot
  be measured on ours. Nor can we tell how many of our ball gaps come from airborne balls: the plan said "crowded and
  aerial", and what was measured is "lines and slow balls" (A08).
- **No positional bias.** The `tracking` profile brings the players' jitter down towards ours (next item) and does nothing else
  about their positional error. A slowly varying bias against the true position cannot be measured on ours (there is no
  reference to measure it against), so none is added. Published accuracies for video-derived positions run from 0.08 to
  0.09 m RMSE for a multi-camera stadium system against a motion-capture reference (Linke, Link and Lames 2020, PLoS ONE
  15(3): e0230179) to a median of 1.1 to 1.7 m (mean 1.8 to 2.8 m) for a pipeline on broadcast video (Theiner et al.,
  WACV 2022, arXiv:2110.11107, Table 3), so a bias of the order of 0.5 to 2 m is plausible for ours and would hurt a
  method that reads distances more than the benchmark says (the plan's range of 0.5 to 2 m is not stated by either
  source, which are the nearest published figures; it is not our measurement).
- **Rougher positions.** On `tracking`, after the 5-frame smoothing the jitter is 1.80 cm per slot against our 3.3 mm,
  5.5 times ours in RMS (T35, R004c). `generic` smooths over 7 frames by a declared rule, and its jitter is not set
  against ours. DFL's clean 9.0 cm is a few jumps on a 1 cm grid, and the average takes it down to a fifth; what
  is left is DFL's, lowered. A method that reads positions, speeds or accelerations sees rougher ones than ours, so
  compare medians and percentiles and not an RMS (T35).
- **`tracking`'s ball gaps (the default until R030) are transplanted with their pattern and not their clock, and are
  blind to the play.** The
  offset is drawn uniformly, so the pattern lags by up to M - n frames (our half less the benchmark's), 7.6 minutes in
  a 68,000-frame half (R003's offsets reached 11,222 frames), and most of the climb through each half (A02) arrives
  shifted. Separately, the mask is cut from our match and laid over DFL's play: a dead ball is no likelier to be missing
  than any other frame, where in the public window ours is about three times as likely (33.6% of the dead frames
  against 11.8% of all of them, A01), and 36.0% of the gaps start within 2 m of a line against 28.0% of ours (R003):
  placed without regard to the play, they start near a line more often in a DFL match than ours do, and DFL tracks the
  ball out of play (R005). `tracking` has no gate on that share, which is `context`'s, and 36.0% is outside the +-0.05
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
  lands in the slot they had 47.4% of the time (529 of 1,117 returns; 48.1% of 1,147 in version 2, 69.0% in version 1,
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
