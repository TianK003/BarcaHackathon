# The P01 benchmark

We may not hand-label our own match, not even to validate (the organisers' clarification of 22 Oct 2026,
[CLAUDE.md](../../CLAUDE.md)), and the public window holds 17 possessions and ranks the baselines the wrong way round
([T01](../../docs/traps.md)). So the choices that matter (decoder, evidence weights, thresholds) are validated on **other
matches**: the seven open DFL/IDSSE Bundesliga matches (CC BY 4.0), which carry the ball's own alive/dead flag and the
team in possession on every frame. A config runs on them through the pipeline's own functions, is scored the way the
organisers' scorer scores, and is compared with another config with the uncertainty that seven matches allow.
The plan is [P01](../../docs/proposals.md); what was measured is in [results.md](../../docs/results.md) (A05 to A07,
R002).

```
python pipeline/benchmark/sources.py          # data -> data/external/dfl/   (21 files, 2.6 GB, size and md5 checked)
python pipeline/benchmark/run.py CONFIG ...   # configs -> scores, intervals, comparisons, one JSON report per config
python pipeline/benchmark/selftest.py [--dfl] # the checks
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
| `--profile P` | how the tracking is made. Only `clean` (DFL's tracking as it is) exists so far; the degraded ones come later. |
| `--seeds N` | degradation seeds per match (default 3). The clean profile has nothing random to repeat, so it always runs seed 0 only; if you pass `--seeds` with more than 1 it says that it ignored it, and says nothing when you pass none. |
| `--matches ID ...` | the matches to run, in this order (default: all seven). Fewer than seven are filed under a name that says which (see the report, below). |
| `--truth raw\|coarse` | the truth to score against (below). Default `raw`. |
| `--data-dir D` | the data bundle (default `data/`, or `$MTAH_DATA_DIR`). |
| `--no-write` | print only. |

A run takes about 10 s on the clean profile with the caches warm (one cached load per match, a second or so per config).
Everything that can be refused is refused before the first match is loaded, with a sentence that names what to change:
no config, a config that does not load, `scope = "file"` (which decodes across half time; only
`pipeline/configs/smooth50_repro.toml` needs it), two configs with the same name, an unknown profile, truth or match.
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

### What the clean profile cannot show

The DFL ball is tracked on every frame, so `nearest_player` votes everywhere and `majority_vote`, which takes `D` only
where nobody voted, labels almost nothing `D`, where the truth has 41% to 52% of every match dead. So on `clean`
the alive/dead F1 is the same for every yardstick (it is the F1 of "always alive", the scorer's ALIVE being the positive
class), the macroF1 is held down by an empty `D` class, and the ranking is only about who has the ball. That is
expected, and is why the degraded profiles (which give the ball our tracking's gaps) exist. It is not a bug of the
configs.

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
| `matches` | per match, in the order run: `id`, `frames`, `graded_frames`, `fps`, `carried_frames` (frames the truth had to carry: 0), and `sources`: the size and sha256 of the two DFL files it was made from (null if the match was not read through the cache, with `sources_reason`) |
| `git`, `environment` | the commit and whether the tree was dirty; the Python, numpy and lxml versions and the platform |
| `pooled`, `per_match` | the scores of all halves, and of each match: `spell_f1_mean`, `spell_f1` at each IoU, `macro_f1`, `alive_dead_f1`, `n_gt`, `n_pred`, `halves` |
| `t_interval` | the t interval of the per-match `spell_f1_mean` and `macro_f1`: `n` (the matches), `level` (0.9), `mean`, `low`, `high`. Only `low` and `high` are null, for fewer than two matches; the `mean` of one match is its own score |
| `possessions_per_graded_minute`, `labels` | the rates and the label shares of the printout |
| `halves` | the counts everything above was made from, per half: `n_gt`, `n_pred`, `tp` at each IoU, the 3x4 `confusion` of the unfilled codes |
| `versus` | null for the reference; for the others, the comparison with it for `spell_f1_mean` and `macro_f1` (`score.compare`: deltas, interval, p, wins, losses, by competition, and whether the interval excludes 0) and the reference's name and config hash. It is the one field that depends on which other configs were run |
| `calibration` | null: the clean profile degrades nothing, so it measures nothing of our tracking |

## The checks

```
.venv/Scripts/python pipeline/benchmark/selftest.py          # synthetic, seconds
.venv/Scripts/python pipeline/benchmark/selftest.py --dfl    # + the seven downloaded matches: 99 to 221 s measured with the caches warm
```

(The first time, while the seven caches are built, the `--dfl` run took 532 s; the machine's load decides the rest.)

`checks/run.py` holds the runner's. The one that anchors the rest is `test_dfl_oracle_scores_perfectly`: an evidence
source that is the truth itself, decoded by `majority_vote` with half-width 0, must score exactly 1.0 on spellF1,
macroF1 and alive/dead F1 on every one of the seven matches.

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

*A stub: the final version is written once the degraded profiles and the fidelity report exist.*

- Seven matches, five of them one club's home games, are a small and not independent sample; the intervals are wide for
  that reason and are no more than that.
- DFL's flags and Metrica's corrected tracking are different annotation systems, and DFL's grain is finer than ours as
  far as we can tell (A06). That is what `--truth coarse` is for, and it does not correct a measured difference.
- The benchmark has no video and no tagging: whatever uses the homography, the Smart Tagging, the camera or the audio is
  checked on the public window and not here.
- The clean profile has the ball on every frame. Until a degraded profile is run, a score here says how a method works
  when it can see the ball.
