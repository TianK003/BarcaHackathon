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
    tracking  the gaps' ball, exactly, and the players broken up the way ours are: tracked spans and lost spans, and
              46 anonymous slots that tracklets take and leave; a wrong-team slot now and then, NaN speeds in runs and
              positions smoothed (below). Not yet our frames with few players tracked, which is the profile's last
              part: three of its fidelity gates, the ones about how many slots are present, are pending until then
              (fidelity.PENDING_GATES, R004).

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

The tracklets (tracking). In ours a slot is not a player: it is a tracklet, a stretch of one slot being present, which
ends when the tracker loses the player and is taken over by whoever it finds next (T15). DFL has one column per player.
The degrader breaks each player's time on the pitch into tracked spans and lost spans and gives each tracked span (a
tracklet) a slot, so that the slots appear, vanish and are reused. A slot's index is still not as empty of the player as
in ours: under the lowest-free rule (below) a player who comes back after a lost span lands in the slot he had 69.0% of
the time (860 of 1,247 returns, J03WMX, seed 0, with the wrong-team episodes on, as in version 1), and a slot idles
between two tracklets for a median of 2.3 s (59.7% of its gaps last at most 3 s and 3.8% over 60 s), where ours idles
for 7.4 s (17.7% at most 3 s, 15.0% over 60 s: A09, R004). Before the episodes were added it was 76.0%, 2.0 s, 67.1% and
none over 60 s (R003): the same code with the scale at 0 gives those again. No evidence source reads a slot's identity
today, beyond nearest_player's argmin on exact ties, which goes to the lower slot; but a method that follows a slot
across a loss would be validated optimistically here. The tail's part of the profile (the frames with few players
tracked) re-measures this.

* What it draws from is ours, measured on our kick-off spans (Calibration): a tracked span's length is one of our
  tracklet lengths (tracklet_lengths_s, in frames at the match's rate, at least 1), drawn uniformly, so that their
  distribution is ours and a span of 1 frame happens as often as in ours; and the lost spans are such that the share of
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
  team's pool; a slot is free for a tracklet that starts at t when its previous tracklet ended before t, which is when
  that tracklet's half-open stop is at most t. A pool with no free slot is a ValueError (a team has 11 players on the
  pitch and 22 or 24 slots, so it means that DFL gave positions to players who are not playing).
* The players cover each graded block, from its first row: not from the kick-off as the ball's mask does, because the
  mask's clock is a property of the gaps (coverage worsens through a half) and a player is tracked or lost whatever the
  clock. A DFL half starts at its kick-off, so the two are the same rows for the 7 matches. The rows between the halves
  have no players, as in DFL.
* The speed is a finite difference in m/s on the match's pitch: the step from the frame before, over the frame's
  duration, within a tracklet piece, and NaN on its first frame (also where another player's tracklet took the slot on
  the frame it was freed: a step between two players is no speed). DFL's own speed channel is not used. Then the NaN
  runs of ours are placed (below).
* The ball is exactly the gaps profile's: the same transplant draws. The profiles nest, so the difference between two
  of their results is one component.

The wrong-team episodes (T32). On 10.5% of our frames a team shows more than 11 slots while the other shows fewer: a
slot of one team carries a player of the other, because the tracker's identity followed the wrong player out of a duel.
Identity swaps happen where players come close, so the degrader cuts the walk's tracked spans with episodes of that
kind (episode_pieces), between the walk and the slot assignment: the walk's "tracklets" draws are not touched, and
the order of operations of a half is the tracklets, the episodes, the slot of every piece, the positions (smoothed),
the speeds, the NaN runs.

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

The NaN-speed runs (place_speed_nan): ours come in runs (7.8% of the present slot-frames, median 21 frames, p99 2,300).
After the speeds, runs are placed from the half's "speed_nan" stream: a present slot-frame chosen uniformly, a length
from our own runs, the next L frames of that slot NaN, stopping at its tracklet's end, until the half's NaN speeds
(the first rows of the tracklets, which are NaN already, included) reach Calibration.speed_nan_share. The ball's speed
stays NaN.

Smoothing. DFL's positions are rougher than ours (SMOOTH_WINDOW: the second difference is 27 times ours), so each
tracklet piece is smoothed with a centred moving average of that many frames, the window shrinking at the ends of the
piece to the rows inside it (smoothed); nothing crosses a piece. What is not matched, and cannot be on ours: a slowly
varying bias against the true position (README, limitations).

The record of what was drawn (info["players"]) holds counts and shares: the parameters (d, the mean tracked and lost
span, the scale, the targets, the smoothing window), per half the tracklets, the slots used, the frames on the pitch and
the lost ones, the episodes' counts and the NaN runs', and their totals. No frame.
"""

import hashlib
import math
import operator
from dataclasses import dataclass, field

import numpy as np

from pipeline.benchmark import calibration as calibration_module
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import sources
from pipeline.inputs.atd_cache import Atd

# Params
CLEAN = "clean"
GAPS = "gaps"
TRACKING = "tracking"
PROFILES = {CLEAN: 1, GAPS: 1, TRACKING: 1}     # name -> version of how its tracking is made (the report says it)
# The streams of random numbers, by the part of the degradation that draws them. The number is the stream's place in the
# key of rng_for, so it never changes for a component and a new component takes the next one. The transplant, the
# tracklets, the wrong-team episodes and the NaN-speed runs are drawn so far; the hazard's is the context profile's.
COMPONENTS = {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4}
TAIL_FRAMES = 7500                  # a mask that runs out goes round the last 300 s of the source (25 Hz)
INDEX_BYTES = 4                     # an id outside the registry has the first 4 bytes of its sha256 as match_index
CAP = 18                            # d is taken over the frames with MORE than this many slots present (not the tail)
FULL = calibration_module.FULL_COUNT    # 22: players on the pitch, so slots present on a frame where nobody is lost
HOME_SLOTS, AWAY_SLOTS = 24, 22     # our tracker's slots (the ATD metadata): the home team's first, then the away's
N_SLOTS = HOME_SLOTS + AWAY_SLOTS   # 46
DECIMALS = 9                        # the floats of a record are rounded to this, as calibration.py's are
TRACKING_FIELDS = ("present_counts", "tracklet_lengths_s", "wrong_team_episodes_s", "wrong_team_share",
                   "speed_nan_runs", "speed_nan_share")         # what the tracking profile reads of the Calibration
NEAR_PAIR_M = 2.0                   # an episode starts only where an opposing tracked pair is this close (inclusive)
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
    n, offset = _whole("n", n), _whole("offset", offset)
    length = mask.size
    tail = min(TAIL_FRAMES, length)
    index = offset + np.arange(n, dtype=np.int64)
    beyond = index >= length
    index[beyond] = length - tail + (index[beyond] - length) % tail
    return mask[index]


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

    The rest is each part's own, and neutral by default (none of the new parts, which is Task 12's profile): `episodes`
    are the wrong-team episodes of ours in frames (`mean_episode` their mean), `wrong_team_share` the share of frames
    that ours has a team in surplus on (the target), and `scale` is s: the onsets come at the rate
    onset_rate = s * wrong_team_share / mean_episode per row (0 with no scale, no share or no episodes). `nan_runs` are
    our runs of NaN speeds in frames and `speed_nan_share` the share of the present slot-frames to reach. `smooth` is
    the window in frames of the moving average over each tracklet piece (0 is none; otherwise odd, at least 3).
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

    @property
    def onset_rate(self):
        """The chance that a row of the half starts an episode: s * target / mean episode frames, 0 if there is none."""
        return self.scale * self.wrong_team_share / self.mean_episode if self.episodes.size else 0.0


@dataclass(frozen=True)
class Tracklet:
    """A stretch of one player being tracked as one slot: rows start .. stop - 1 of the match (half-open)."""
    player: int         # the column of the DFL match
    start: int
    stop: int
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
                        speed_nan_share=0.0, smooth=0):
    """TrackingParameters of tracked spans `lengths` (whole frames, each at least 1) and a lost share d (0 to below 1).

    mu = mean_tracked * d / (1 - d). A lost span is at least 1 frame, so a mean below 1 cannot be drawn: a d so small
    that mu comes out below 1 (and above 0) is a ValueError that says so. d = 0 has mu = 0 and loses nobody.

    The other arguments are the parts that come after the tracklets, and are none by default. `episodes` (whole frames,
    each at least 1: wrong-team episodes of ours) and `wrong_team_share` (0 to below 1) with the `scale` s (at least 0)
    give the onsets' rate s * share / mean episode, which must be below 1; a share and a scale above 0 need episodes.
    `nan_runs` (whole frames, each at least 1) and `speed_nan_share` (0 to below MAX_SPEED_NAN_SHARE) are the NaN-speed
    runs and the share to reach, which needs runs. `smooth` is 0, or the odd window (at least 3) of the moving average.
    ValueError, naming the argument, for anything else.
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
                              speed_nan_share=nan_share, smooth=int(smooth))


def calibrated_parameters(calibration, fps, profile=TRACKING):
    """TrackingParameters from the Calibration: its tracklet lengths and episodes at the match's rate, d from its slots.

    A tracked span is max(1, round(seconds * fps)) frames, `fps` the match's own (DFL's 25); d is lost_share over
    present_counts. A wrong-team episode is converted the same way (wrong_team_episodes_s, both directions together),
    and the mean episode is the mean of those very lengths; the NaN-speed runs are already in frames
    (speed_nan_runs). The targets are the Calibration's wrong_team_share and speed_nan_share, the scale is
    WRONG_TEAM_SCALE and the smoothing window SMOOTH_WINDOW. ValueError, naming the profile, for a Calibration that
    lacks any of TRACKING_FIELDS (the made-up ones of the checks, a Calibration of an older layout), for no tracklets,
    for a share above 0 with no episodes or runs to draw, and for an fps that is not a whole number, at least 1.
    """
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
                               smooth=SMOOTH_WINDOW)


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


def assign_slots(view, half, pieces):
    """[Tracklet] of the pieces (first, player, last, pool), in order of (start, player), each in the lowest free slot.

    Each piece takes the lowest slot of its pool's team (home 0-23, away 24-45) that its previous tracklet has left: a
    slot is free from the stop of its last tracklet, and a tracklet that starts on that very row can take it.
    ValueError, naming the match, the half and the row, for a pool with no free slot.
    """
    free_from = [0] * N_SLOTS
    tracklets = []
    for first, player, last, pool in pieces:
        home = pool == 0
        slots = range(HOME_SLOTS) if home else range(HOME_SLOTS, N_SLOTS)
        slot = next((candidate for candidate in slots if free_from[candidate] <= first), None)
        if slot is None:
            raise ValueError("{}: half {}: no free {} slot for the player in column {}, whose tracklet starts at frame "
                             "{}: the {} slots of its pool are all taken by players tracked at that frame. A team has "
                             "11 players on the pitch, so look at where the match's players_xy gives a position to a "
                             "player who is not playing".format(view.match_id, half, "home" if home else "away", player,
                                                                first, len(slots)))
        free_from[slot] = last
        tracklets.append(Tracklet(player=player, start=first, stop=last, slot=slot))
    return tracklets


@dataclass(frozen=True, eq=False)
class HalfPlayers:
    """One half's players as slots: the tracklets in order of (start, player) and the episodes' counts (counts only)."""
    tracklets: list
    counts: dict


def choose_half(view, half, parameters, seed):
    """HalfPlayers of one half: the walk's spans (walk_spans), cut by the wrong-team episodes (episode_pieces), slotted.

    The order is the one the degradation documents: the per-player tracklets, the episodes that split them, then the
    slot of every piece (assign_slots). ValueError as walk_spans and assign_slots.
    """
    pieces, counts = episode_pieces(view, half, walk_spans(view, half, parameters, seed), parameters, seed)
    return HalfPlayers(tracklets=assign_slots(view, half, pieces), counts=counts)


def choose_tracklets(view, half, parameters, seed):
    """[Tracklet] of one half of a match, in order of (start, player), each with the lowest free slot of its pool.

    The tracklets of choose_half: with no wrong-team episodes (the parameters' defaults) these are Task 12's.
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


def degraded_players(view, parameters, seed):
    """(players_xy, players_speed, tracklets, counts): the match's players as slots, the tracklets {half: [Tracklet]}
    and the counts {half: {...}} of what was drawn for each half (_players_record).

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
        chosen = choose_half(view, half, parameters, seed)
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
                          "speed_nan_frames"), 0)
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
            "speed_nan_target": round(parameters.speed_nan_share, DECIMALS), **sums,
            "speed_nan_share": (round(sums["speed_nan_frames"] / (on_pitch - lost), DECIMALS)
                                if on_pitch > lost else None),
            "halves": {half: halves[half] for half in sorted(halves)}}


# -- the profiles -----------------------------------------------------------------------------------------------------

def degrade(view, profile, calibration, seed):
    """(Atd, info): the tracking of `view` as `profile` makes it under `seed`, and what was drawn to make it.

    `calibration` is the Calibration of BAR-ATH (calibration.load()): the gaps and tracking profiles need it, the clean
    one does not use it. The Atd has the BAR-ATH cache's own layout (inputs.clean_atd), every array read-only, and the
    length of the match; the tracking profile's has 46 slots (slot_team) where the others have one per DFL player.
    `info` is JSON-ready and holds no frame of anybody: {"profile", "version", "seed", "match_id", "match_index"} and,
    for gaps and tracking, "transplants" ({half: {"source_half", "offset", "frames"}}) and "mask_share": the share of
    the frames that the transplant covers where the transplanted mask says the ball is missing. It is the mask's
    share, not the ball's after it: a ball that DFL itself has no frame for is missing as well. For tracking also
    "players": the parameters and the counts of the tracklets, episodes and NaN runs (_players_record). ValueError for
    a profile that is not in PROFILES, for a seed that is not a whole number of at least 0, for a profile that needs
    the Calibration without one (or, for tracking, without the fields it reads: calibrated_parameters), and for a pool
    of slots with no free slot (choose_tracklets).
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
                         "calibration.load()".format(profile, " and its tracklets" if profile == TRACKING else ""))
    ball_xy, ball_speed, info["transplants"], info["mask_share"] = _gap_ball(view, atd, calibration, seed)
    if profile == GAPS:
        players = (atd.players_xy, atd.players_speed, atd.slot_team)
    else:
        parameters = calibrated_parameters(calibration, view.fps, profile)
        xy, speed, records, counts = degraded_players(view, parameters, seed)
        info["players"] = _players_record(view, parameters, records, counts)
        players = (xy, speed, slot_team())
    degraded = Atd(players_xy=players[0], players_speed=players[1], ball_xy=ball_xy, ball_speed=ball_speed,
                   slot_team=players[2])
    return degraded, info


def _gap_ball(view, atd, calibration, seed):
    """(ball_xy, ball_speed, transplants, mask_share): the clean ball with our missing-ball masks transplanted onto it.

    Shared by the gaps profile and every profile that adds to it, so that their balls are the same bytes: the transplant
    draws are the "transplant" component's, whatever else a profile draws. The arrays are read-only."""
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
    return (ball_xy, ball_speed, {half: transplants[half] for half in sorted(transplants)},
            missing / covered if covered else None)
