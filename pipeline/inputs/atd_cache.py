"""Parse the raw tactical ATD once and keep it as numpy arrays under <data dir>/cache/atd_tactical/.

Streaming the 195 MB text file costs starter/baseline.py a minute or two on every run (docs/traps.md T19). Here it
is read once, checked, and stored as .npy arrays plus a manifest.json; every later run loads the arrays in about a
second. The cache lives under the data directory, which git ignores -- frame-level arrays derived from the match
must never be committed (CLAUDE.md, hard rules).

The broadcast's ATD (R028) is the same format with its own slots and frames, so the same code parses and checks it,
into a folder of its own (load_or_build's `folder`: atd_broadcast/, pipeline/inputs/__init__.py).

Exactness
---------
The parser maps the two separators to commas and hands the file to pandas' C reader with
float_precision="round_trip", which uses Python's own string-to-float conversion. On the delivered file that gives
bit-identical float64s to the float() calls in starter/baseline.py on all 28.3 million values (measured
2026-10-22). Anything less and a near-tie between two players could resolve differently, and the regression against
the committed baselines could not be byte-exact.

Freshness
---------
The manifest records the sha256 and size of both source files, the cache format version and the numpy/pandas
versions that parsed them. A cache is reused only if all of those still match; otherwise it is rebuilt, which is
what keeps a re-downloaded or replaced bundle from being read through a stale cache. Hashing costs about a second
per run, which is the price of never having to wonder.
"""

import functools
import hashlib
import io
import json
import os
import platform
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from starter import baseline

# Params
CACHE_VERSION = 1      # bump when the arrays or their meaning change; every existing cache then rebuilds
FOLDER = "atd_tactical"
MANIFEST = "manifest.json"
ARRAYS = ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team")
FIELDS = 3             # x, y, speed for every slot and for the ball
# Raw line: "<frame_id>:<slot>;<slot>;...;<slot>:<ball>", each group "x,y,speed". Mapping both separators to commas
# turns every line into one flat CSV row: frame_id, then S groups of three, then the ball's three.
TO_COMMAS = bytes.maketrans(b":;", b",,")
HASH_CHUNK = 1 << 20


@dataclass(frozen=True, eq=False)
class Atd:
    """The tracking, one row per submission row (row i = ATD frame_id i + 1 = tactical video frame i).

    Coordinates are normalised 0-1, exactly as in the file: the homography takes them as they are, and distances
    need them scaled by the pitch first (Inputs.metres, docs/traps.md T16). NaN marks an absent slot or ball. A slot
    is a tracklet, not a player: 46 of them for 22 players, appearing, vanishing and swapping (T15).

    Every array is read-only. A component that wants to transform one makes a copy; that is what stops one
    evidence source from silently changing the input of the next.

    No array is in its repr (repr=False on every field): numpy prints the head and tail of an array, and an Atd printed
    by a log line, an assertion message or a debugger would put frames of the match's tracking in a log (and, for the
    benchmark's degraded Atd, show where the ball was removed). Print its shapes, or count.
    """
    players_xy: np.ndarray = field(repr=False)     # (n, S, 2) float64
    players_speed: np.ndarray = field(repr=False)  # (n, S) float64 -- may be NaN while the position exists (README)
    ball_xy: np.ndarray = field(repr=False)        # (n, 2) float64
    ball_speed: np.ndarray = field(repr=False)     # (n,) float64 -- same caveat
    slot_team: np.ndarray = field(repr=False)      # (S,) int8: 0 = home (Barcelona), 1 = away, from the metadata XML

    @property
    def n_frames(self):
        return len(self.ball_xy)

    @property
    def n_slots(self):
        return len(self.slot_team)

    @functools.cached_property
    def players_present(self):
        """(n, S) bool. Present means x is a number; x and y are always missing together, which parse() checks."""
        return _read_only(~np.isnan(self.players_xy[:, :, 0]))

    @functools.cached_property
    def ball_present(self):
        """(n,) bool. The ball is missing on 41% of graded frames, in long gaps (docs/results.md A01)."""
        return _read_only(~np.isnan(self.ball_xy[:, 0]))


def load_or_build(meta_path, raw_path, cache_dir, n_frames, rebuild=False, folder=FOLDER):
    """(Atd, manifest, built) -- from the cache when it is fresh, otherwise parsed from the raw file and stored.

    built is True when this call parsed the raw file. rebuild=True forces that. The cache is <cache_dir>/<folder>/:
    atd_tactical/ for ours (FOLDER), and one folder per other ATD, so that two never share a manifest.
    """
    folder = os.path.join(cache_dir, folder)
    key = {
        "cache_version": CACHE_VERSION,
        "n_frames": n_frames,
        "sources": {"atd_meta": describe_source(meta_path), "atd_raw": describe_source(raw_path)},
        "parsed_with": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest = None if rebuild else _read_manifest(folder)
    if manifest is not None and all(manifest.get(name) == value for name, value in key.items()):
        try:
            return _load(folder), manifest, False
        except (OSError, ValueError, EOFError):
            pass  # a manifest whose arrays were deleted or truncated: rebuild below rather than fail

    atd = parse(raw_path, baseline.team_of_each_track(meta_path), n_frames)
    manifest = dict(key, n_slots=atd.n_slots, python=platform.python_version(), stats=describe(atd))
    _save(atd, folder, manifest)
    return atd, manifest, True


def parse(raw_path, slot_team, n_frames):
    """Read the raw ATD into an Atd, checking its structure before trusting a single position.

    slot_team comes from starter/baseline.team_of_each_track, so the home/away convention (first <Team> in the
    metadata is home) is the organisers' own and lives in one place.
    """
    slot_team = np.asarray(slot_team, dtype=np.int8)
    n_slots = len(slot_team)
    with open(raw_path, "rb") as handle:
        blob = handle.read()
    _check_structure(blob, n_slots, n_frames, raw_path)

    table = pd.read_csv(io.BytesIO(blob.translate(TO_COMMAS)), header=None, dtype=np.float64,
                        float_precision="round_trip", engine="c").to_numpy()
    del blob
    if table.shape != (n_frames, 1 + FIELDS * n_slots + FIELDS):
        raise ValueError("{}: parsed {} where {} rows x {} columns were expected".format(
            raw_path, table.shape, n_frames, 1 + FIELDS * n_slots + FIELDS))
    # The whole alignment story of the ATD (CLAUDE.md): frame_id runs 1..n, and row i is video frame i. Checked, not
    # assumed, because the off-by-one here would shift every transition of the match (docs/traps.md T08).
    if not np.array_equal(table[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame_id is not 1..{} in order".format(raw_path, n_frames))

    players = table[:, 1:1 + FIELDS * n_slots].reshape(n_frames, n_slots, FIELDS)
    ball = table[:, 1 + FIELDS * n_slots:]
    atd = Atd(players_xy=np.ascontiguousarray(players[:, :, :2]),
              players_speed=np.ascontiguousarray(players[:, :, 2]),
              ball_xy=np.ascontiguousarray(ball[:, :2]),
              ball_speed=np.ascontiguousarray(ball[:, 2]),
              slot_team=slot_team.copy())
    _check_presence(atd, raw_path)
    return _freeze(atd)


def describe(atd):
    """Aggregates worth keeping in the manifest. Counts and shares only -- nothing that could rebuild a frame."""
    present = atd.players_present
    per_frame = present.sum(axis=1)
    return {
        "ball_present_share": round(float(atd.ball_present.mean()), 6),
        "slots_present_per_frame": {"min": int(per_frame.min()), "median": float(np.median(per_frame)),
                                    "max": int(per_frame.max())},
        "speed_missing_while_present": {
            "slot_frames": int((np.isnan(atd.players_speed) & present).sum()),
            "ball_frames": int((np.isnan(atd.ball_speed) & atd.ball_present).sum()),
        },
    }


def describe_source(path):
    """{"bytes", "sha256"} of a file, read in chunks so a 195 MB file never sits in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return {"bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def _check_structure(blob, n_slots, n_frames, raw_path):
    """Exact separator counts, before parsing.

    pandas pads a row that is *short* with NaN at the end, so a line missing one slot group would shift the ball's
    columns and read silently as "ball not tracked". A row that is too long raises inside pandas. Counting every
    separator over the whole file closes the first hole: with these exact totals, and no line allowed to be longer,
    every line has exactly the expected shape.
    """
    lines = blob.count(b"\n") + (0 if blob.endswith(b"\n") else 1)
    expected = {"lines": n_frames, "':'": 2 * n_frames, "';'": (n_slots - 1) * n_frames,
                "','": (2 * n_slots + 2) * n_frames}
    found = {"lines": lines, "':'": blob.count(b":"), "';'": blob.count(b";"), "','": blob.count(b",")}
    if found != expected:
        raise ValueError(
            "{}: expected {} lines of '<frame_id>:<{} slots>:<ball>' with x,y,speed groups, but the separator "
            "counts are {} where that format needs {}".format(raw_path, n_frames, n_slots, found, expected))


def _check_presence(atd, raw_path):
    """x and y must be missing together, for every slot and for the ball.

    Everything downstream -- and starter/baseline.py, which tests only the first character of a group -- treats a
    position as present when x is. On the delivered file there are 0 exceptions; if a new export has some, decide
    what a half-missing position means before relaxing this.
    """
    slots = int((np.isnan(atd.players_xy[:, :, 0]) != np.isnan(atd.players_xy[:, :, 1])).sum())
    balls = int((np.isnan(atd.ball_xy[:, 0]) != np.isnan(atd.ball_xy[:, 1])).sum())
    if slots or balls:
        raise ValueError("{}: x and y are not missing together on {} slot-frames and {} ball frames".format(
            raw_path, slots, balls))


def _read_manifest(folder):
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save(atd, folder, manifest):
    """Arrays first, manifest last, with the old manifest removed up front.

    The manifest is what marks a cache as complete, so a run killed half-way through writing leaves none, and the
    next run rebuilds instead of loading a mixture of old and new arrays.
    """
    os.makedirs(folder, exist_ok=True)
    manifest_path = os.path.join(folder, MANIFEST)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    for name in ARRAYS:
        np.save(os.path.join(folder, name + ".npy"), getattr(atd, name), allow_pickle=False)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load(folder):
    # Loaded into memory rather than memory-mapped: Windows locks a mapped file, which would make the next rebuild,
    # or a temporary folder's cleanup, fail for as long as anything still held an array.
    return _freeze(Atd(**{name: np.load(os.path.join(folder, name + ".npy"), allow_pickle=False)
                          for name in ARRAYS}))


def _freeze(atd):
    for name in ARRAYS:
        _read_only(getattr(atd, name))
    return atd


def _read_only(array):
    array.flags.writeable = False
    return array
