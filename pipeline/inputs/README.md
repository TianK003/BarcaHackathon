# Stage 1 — inputs

**The point of this stage:** every later stage works on arrays where row *i* is frame *i* of the tactical video —
the submission's own index. The data does not arrive that way (the ATD counts from 1, Smart Tagging in seconds, the
broadcast on a different timeline), and one off-by-one shifts every transition of the match. So the conversions
happen once, here, are checked, and nothing downstream ever converts a frame index again.

The second job is speed: the raw ATD is 195 MB of text, which `starter/baseline.py` streams for 1–2 minutes on every
run. Here it is parsed once into a cache and reloaded in about a second.

## In → out

| in (`data/tactical/`) | out (`Inputs`) |
|---|---|
| `BAR-ATH_tactical_atd.txt` + `_atd.xml` | `inputs.atd` — `Atd` arrays, see below |
| `BAR-ATH_tactical_homography.csv` | `inputs.homography` — `matrices (n, 3, 3)`, `error (n,)`, `valid (n,)` |
| `BAR-ATH_tactical_smart_tagging.xml` | `inputs.smart_tagging` — list of `{code, tags, start_frame, end_frame, …}` |
| `games.py` (constants, no file) | `n_frames` 199,319 · `fps` 25 · `pitch` (105.3, 68.0) m · `graded` · `half_of_frame` · `graded_mask` · `kickoffs` |
| `data/broadcast/BAR-ATH_broadcast_atd.txt` + `_atd.xml` | `inputs.broadcast_atd` — an `Atd` on the **broadcast's** timeline (164,488 rows, 51 slots) |
| `data/broadcast/BAR-ATH_broadcast_homography.csv` | `inputs.broadcast_valid` — `(164,488,)` bool, its `valid` column only |
| both, aligned by code ([below](#the-broadcasts-ball-r028)) | `inputs.broadcast_alignment`, `inputs.broadcast_ball` `(n, 2)`, `inputs.broadcast_summary` |
| `[inputs] ball_fill` of the config | `inputs.with_ball_fill(kind)` — the inputs themselves, or a `FilledInputs` whose `atd` has the broadcast's ball in our gaps · `ball_filled` `(n,)` · `fill_summary` |

Everything is **lazy**: `Inputs()` reads nothing, and an asset is loaded the first time someone asks for it. A run
that never touches the homography never pays for it, and nothing of the broadcast is read unless a config fills the
ball.

## Alignment — done here, nowhere else

| source | its index | → row | where |
|---|---|---|---|
| ATD | `frame_id` 1…199,319 | `frame_id - 1` | `atd_cache.parse` (checks ids are exactly 1…n in order) |
| Homography | `frame` 0…199,318 | as is | `Inputs.homography` (checks row count and frame column) |
| Smart Tagging | seconds from the first frame | `round(s * 25)` | `starter/load_smart_tagging.load` |
| Broadcast | its own 164,488-frame timeline | row *t* from broadcast frame *t − d*, one *d* per half, **found by code** on every run | `broadcast.align`, `broadcast.frame_map`; only `broadcast_ball` is on our rows — see [T09](../../docs/traps.md) |

Graded frames (`inputs.graded`, half-open): 1st half `11,185–90,651`, 2nd half `110,808–190,197`. The 2nd half is
graded from 2,934 frames before its kick-off at 113,742, and all of those are `D` ([T02](../../docs/traps.md)).

## The `Atd` arrays

| array | shape | meaning |
|---|---|---|
| `players_xy` | (n, 46, 2) float64 | normalised 0–1 position per track slot; NaN = slot absent |
| `players_speed` | (n, 46) float64 | speed channel per slot; can be NaN while the position exists |
| `ball_xy` | (n, 2) float64 | normalised ball position; NaN = ball not tracked |
| `ball_speed` | (n,) float64 | same caveat as the players' speed |
| `slot_team` | (46,) int8 | 0 = home (Barcelona), 1 = away, from the metadata XML |
| `players_present`, `ball_present` | (n, 46), (n,) bool | derived: x is not NaN |

- **Normalised coordinates**, as in the file, because the homography takes them unconverted. Use
  `inputs.metres(xy)` before any distance: the pitch is 105.3 × 68.0 m, not square ([T16](../../docs/traps.md)).
- **A slot is a tracklet, not a player** — 46 slots for 22 players, appearing, vanishing, swapping and occasionally
  duplicating someone. Build permutation-invariant features ([T15](../../docs/traps.md)).
- **All arrays are read-only.** `atd.players_xy *= 105.3` raises instead of quietly changing the input of every
  source that runs after yours. Copy first if you need to transform.

## Facts about the tracking (measured 2026-10-22, aggregates only)

- The ball is missing on 41% of graded frames, in long gaps (median 10.7 s) — [A01](../../docs/results.md).
- 1 to 35 slots are present per frame.
- x and y are always missing together, for every slot and the ball; the parser checks this and refuses a file where
  they are not.
- Speed is NaN on 370,297 slot-frames and 103,579 ball frames **where the position exists**. Presence means "x is a
  number", never "speed is a number" ([T28](../../docs/traps.md)).

## The ATD cache

| | |
|---|---|
| where | `<data dir>/cache/atd_tactical/` — inside `data/`, so git ignores it. Never put a cache anywhere else. The broadcast's ATD, parsed by the same code, has its own: `<data dir>/cache/atd_broadcast/` |
| what | one `.npy` per array + `manifest.json` (source hashes, versions, aggregate statistics) |
| first run | ~15 s: separators → commas, pandas' C parser with `float_precision="round_trip"` |
| later runs | ~1–2 s: hash both source files, compare with the manifest, load the arrays |
| rebuilt when | either source file's bytes change, `CACHE_VERSION` is bumped, numpy/pandas versions change, or `--rebuild-cache` |

Two properties are worth knowing because the regression depends on them:

- **Bit-exact.** The round-trip parser gives the same float64s as Python's `float()` — the way `starter/baseline.py`
  reads the file — on all 28.3 million values. With anything less, a near-tie between two players could resolve
  differently and the baselines could not be rebuilt byte for byte.
- **Structure checked before parsing.** pandas pads a *short* line with NaN, which would silently read as "no ball".
  So the separators are counted first (`:` 2 per line, `;` 45, `,` 94), and the file is refused if any total is
  off.

## The broadcast's ball (R028)

Our ball is missing on 41% of the graded frames. The TV broadcast tracks the same match from a camera a person steers,
and where ours is lost its ball is often there (A20). A config with

```toml
[inputs]
ball_fill = "broadcast"   # default "none": the tactical ball only
```

runs on `inputs.with_ball_fill("broadcast")`, a `FilledInputs` whose `atd` has the broadcast's usable ball wherever
ours is missing on a graded frame. Every source reads it unchanged; `ball_filled` marks the frames written. The steps
are plain numpy functions in [`broadcast.py`](broadcast.py), with no randomness, and its module docstring has the why:

| step | what | constants (`broadcast.py` Params) |
|---|---|---|
| `align` | per half the offset *d* (our frame *t* = broadcast frame *t − d*), ball against ball in metres: coarse every 25th frame and every 25 frames of offset, within 3 m; fine every frame within ±40 of the coarse best, within 1.5 m; an offset counts on 300 pairs or more | accepted only if the fine score ≥ 0.5 and the coarse best ≥ 2 × the best score more than 250 frames away, else `AlignmentError` stops the run; ties to the smaller \|*d*\| |
| `live` | a mapped frame whose broadcast homography is valid, shows ≥ 4 players, and whose players' median distance to our nearest is < 2.0 m | teams do not enter (T32) |
| `usable` | the broadcast's ball present on a live frame, in runs of ≥ 13 frames (shorter runs are a median 16–27 m off, A20 part 6) | a run breaks where the broadcast's frames stop following each other |
| `fill` | `ball_xy` from the broadcast where ours is missing on a graded frame | `ball_speed` stays NaN (T31); a present ball and the players are never touched |

`run.py` logs one `fill` line (the offsets, the fine scores, the live share, the frames filled), and the run manifest's
`inputs` gains `broadcast`: those aggregates, the broadcast ATD cache's sources and the homography file's size and
sha256. Without `[inputs]`, `with_ball_fill("none")` is the inputs object itself and nothing changes. The benchmark has
no broadcast: its `MatchInputs` serves a simulated ball given as `broadcast_ball=`, and the same fill applies to it.

## Adding an asset

1. Add a lazy `functools.cached_property` to `Inputs` that returns arrays with **n_frames rows**, aligned here.
2. If it is slow to load (video frames, audio features, a broadcast alignment), cache it under `self.cache_dir` with a
   manifest keyed by the source file's sha256, the way `atd_cache.py` does.
3. Check the alignment you rely on — row count, first frame, frame order — and raise with a sentence if it fails.
4. Make the arrays read-only, add a row to the tables above, and log any measurement in
   [docs/results.md](../../docs/results.md).
5. If `FilledInputs` should serve it unchanged, add it to its properties read from the base (`_from_base`).

Proposals that will add assets here: P14 (detected ball). P15's broadcast is R028's ball fill, above. P10's audio track
was added and removed again (dropped, R018, R018b; recoverable from `a08e381`). Decoding video must be frame-exact
([T23](../../docs/traps.md)).

## Traps this stage absorbs

[T08](../../docs/traps.md) off-by-one · T09 broadcast timeline · T10 the `offsets` in `games.py` are not for the
ATD · T15 slots are tracklets · T16 units · T17 Smart Tagging quirks · T19 kloppy is slow · T28 speed presence ·
T29 kick-off frames (`inputs.kickoffs` uses the documented 11,185 / 113,742).
