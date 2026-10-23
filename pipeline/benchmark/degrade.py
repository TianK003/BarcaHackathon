"""Make a DFL match's tracking look like ours: the profiles, the view the degrader sees and the random draws.

    view = tracking_view(match)                        # the tracking of a DflMatch, and nothing of its truth
    atd, info = degrade(view, "gaps", calibration, 0)  # an Atd of the BAR-ATH cache's layout, and what was drawn

What this is for. The benchmark's matches have the ball on every frame and one slot per player, and a pipeline
choice validated on that says little about ours: the ball is missing on 41% of our graded frames, in long gaps
(A01, A02), and the pipeline chooses what to do about exactly that. A profile is a recipe for making the DFL tracking
as bad as ours, and fidelity.py measures with the Calibration's own statistics functions how far it got. Profiles:

    clean   DFL's tracking as it is (inputs.clean_atd).
    gaps    the ball is missing where OUR ball was missing: our own missing-ball masks, transplanted (below).
            The ball's speed is NaN everywhere (our speed channel is NaN on 92% of the frames that have a ball: T31).
            The players are untouched.

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
half longer than M - offset) the mask goes on from the start of its last TAIL_FRAMES frames, round and round: the
kick-off clock of the mask is kept as far as the mask goes, and what follows is a stretch of the late half's pattern
repeated, not a hole in the degradation. The clock is the half's own, from its kick-off: coverage worsens through
each half (A02), a benchmark half starts at its kick-off, and the Calibration measures ours from there too.

What the transplant keeps of ours, by construction: the gaps' durations and the share missing, per half and through
the half. What it does not: where the gaps are in the play. They are drawn without looking at the DFL play, so
nothing that ties ours to it is in them: a dead ball is no likelier to be missing than any other frame (in the public
window ours is about three times as likely as a frame of the window in general, 33.6% of the dead frames against 11.8%
of all of them, A01), and where on the pitch a gap starts is the DFL ball's own place at a random frame, not ours (the
share of onsets within 2 m of a line differs from ours, 36.0% against 28.0% in R003; the direction is not the one the
plan guessed, and there is no base rate of ours to say what ours is relative to). That is the difference between this
profile and the `context` one, if it is built, and it is why fidelity.py checks the share of dead frames with the ball
visible and the nearest player's agreement as warnings, and prints the statistic.

The ball is missing where the mask says so, and where DFL has no ball itself (a row with none stays NaN). Position x
and y are removed together, as an Atd requires (atd_cache._check_presence). Every array of a degraded Atd is read-only.
"""

import hashlib
import operator
from dataclasses import dataclass, field

import numpy as np

from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import sources
from pipeline.inputs.atd_cache import Atd

# Params
CLEAN = "clean"
GAPS = "gaps"
PROFILES = {CLEAN: 1, GAPS: 1}      # name -> version of how its tracking is made (the report says it)
# The streams of random numbers, by the part of the degradation that draws them. The number is the stream's place in the
# key of rng_for, so it never changes for a component and a new component takes the next one. Only the transplant is
# drawn in this task; the others are what the profiles that come later will draw from.
COMPONENTS = {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4}
TAIL_FRAMES = 7500                  # a mask that runs out goes round the last 300 s of the source (25 Hz)
INDEX_BYTES = 4                     # an id outside the registry has the first 4 bytes of its sha256 as match_index


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
    n, offset = _whole("n", n), _whole("offset", offset)
    length = mask.size
    tail = min(TAIL_FRAMES, length)
    index = offset + np.arange(n, dtype=np.int64)
    beyond = index >= length
    index[beyond] = length - tail + (index[beyond] - length) % tail
    return mask[index]


def _span(view, half):
    """(kick-off row, stop row) of a half of the view: the frames a degradation covers, as Calibration.ball_masks."""
    for number, start, stop in view.graded:
        if number == half:
            kickoff = view.kickoffs.get(half, start)
            if not start <= kickoff < stop:
                raise ValueError("{}: half {} kicks off at row {}, outside its block {}-{}".format(
                    view.match_id, half, kickoff, start, stop))
            return kickoff, stop
    raise ValueError("{}: no half {} in its graded blocks {}".format(
        view.match_id, half, ", ".join(str(block) for block in view.graded)))


# -- the profiles -----------------------------------------------------------------------------------------------------

def degrade(view, profile, calibration, seed):
    """(Atd, info): the tracking of `view` as `profile` makes it under `seed`, and what was drawn to make it.

    `calibration` is the Calibration of BAR-ATH (calibration.load()): the gaps profile needs it, the clean one does not
    use it. The Atd has the BAR-ATH cache's own layout (inputs.clean_atd), every array read-only, and the length of the
    match. `info` is JSON-ready and holds no frame of anybody: {"profile", "version", "seed", "match_id",
    "match_index"} and, for gaps, "transplants" ({half: {"source_half", "offset", "frames"}}) and "mask_share": the
    share of the frames that the transplant covers where the transplanted mask says the ball is missing. It is the
    mask's share, not the ball's after it: a ball that DFL itself has no frame for is missing as well. ValueError for a
    profile that is not in PROFILES, for a seed that is not a whole number of at least 0 and for gaps without a
    Calibration.
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
        raise ValueError("profile {!r} takes its ball gaps from our own match, so it needs the Calibration; pass "
                         "calibration.load()".format(profile))
    ball_xy = np.array(atd.ball_xy)
    transplants, covered, missing = {}, 0, 0
    for half, _, _ in view.graded:
        kickoff, stop = _span(view, half)
        choice = choose_transplant(view, half, calibration, seed)
        gone = transplant(calibration.ball_masks[choice.source_half], stop - kickoff, choice.offset)
        ball_xy[kickoff:stop][gone] = np.nan            # x and y together, as an Atd requires
        transplants[half] = {"source_half": choice.source_half, "offset": choice.offset, "frames": stop - kickoff}
        covered += stop - kickoff
        missing += int(gone.sum())
    ball_speed = np.full(view.n_frames, np.nan)
    for array in (ball_xy, ball_speed):
        array.setflags(write=False)
    info["transplants"] = {half: transplants[half] for half in sorted(transplants)}
    info["mask_share"] = missing / covered if covered else None
    degraded = Atd(players_xy=atd.players_xy, players_speed=atd.players_speed, ball_xy=ball_xy, ball_speed=ball_speed,
                   slot_team=atd.slot_team)
    return degraded, info
