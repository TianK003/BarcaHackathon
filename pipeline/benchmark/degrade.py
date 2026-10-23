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
              46 anonymous slots that tracklets take and leave (below). Work in progress: this version has none of our
              wrong-team slots, none of our NaN-speed runs and none of our frames with few players tracked, which are
              the profile's next parts, and so it has no fidelity gates and no yardstick yet.

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
tracklet) a slot, so that the slots appear, vanish and are reused, and a slot index means nothing about the player.

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
  duration, within a tracklet, and NaN on its first frame (also where another player's tracklet took the slot on the
  frame it was freed: a step between two players is no speed). DFL's own speed channel is not used. Our NaN-speed runs
  (7.8% of the present slot-frames) are another part's.
* The ball is exactly the gaps profile's: the same transplant draws. The profiles nest, so the difference between two
  of their results is one component.

The record of what was drawn (info["players"]) holds counts and shares: the parameters (d, the mean tracked and lost
span), per half the tracklets, the slots used, the frames on the pitch and the lost ones, and their totals. No frame.
"""

import hashlib
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
# key of rng_for, so it never changes for a component and a new component takes the next one. The transplant and the
# tracklets are drawn so far; the others are what the parts of the tracking profile that come later will draw from.
COMPONENTS = {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4}
TAIL_FRAMES = 7500                  # a mask that runs out goes round the last 300 s of the source (25 Hz)
INDEX_BYTES = 4                     # an id outside the registry has the first 4 bytes of its sha256 as match_index
CAP = 18                            # d is taken over the frames with MORE than this many slots present (not the tail)
FULL = calibration_module.FULL_COUNT    # 22: players on the pitch, so slots present on a frame where nobody is lost
HOME_SLOTS, AWAY_SLOTS = 24, 22     # our tracker's slots (the ATD metadata): the home team's first, then the away's
N_SLOTS = HOME_SLOTS + AWAY_SLOTS   # 46
DECIMALS = 9                        # the floats of a record are rounded to this, as calibration.py's are
TRACKING_FIELDS = ("present_counts", "tracklet_lengths_s")      # what the tracking profile reads of the Calibration


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

@dataclass(frozen=True, eq=False)
class TrackingParameters:
    """What the walk draws from: our tracked spans, in frames, the lost share d and the mean lost span mu.

    `lengths` are the tracked spans of ours (int64, each at least 1, read-only); a tracked span of the degraded match is
    one of them, drawn uniformly. `d` is the share of an on-pitch player's frames that are lost, `mean_tracked` the mean
    of `lengths` and `mu` = mean_tracked * d / (1 - d) the mean lost span (0 for d = 0: nobody is lost).
    """
    lengths: np.ndarray = field(repr=False)
    d: float
    mean_tracked: float
    mu: float


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


def tracking_parameters(lengths, d):
    """TrackingParameters of tracked spans `lengths` (whole frames, each at least 1) and a lost share d (0 to below 1).

    mu = mean_tracked * d / (1 - d). A lost span is at least 1 frame, so a mean below 1 cannot be drawn: a d so small
    that mu comes out below 1 (and above 0) is a ValueError that says so. d = 0 has mu = 0 and loses nobody.
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
    return TrackingParameters(lengths=array, d=float(d), mean_tracked=mean, mu=mu)


def calibrated_parameters(calibration, fps, profile=TRACKING):
    """TrackingParameters from the Calibration: its tracklet lengths at the match's rate, and d from its slot counts.

    A tracked span is max(1, round(seconds * fps)) frames, `fps` the match's own (DFL's 25); d is lost_share over
    present_counts. ValueError, naming the profile, for a Calibration that lacks either field (the made-up ones of the
    checks, a Calibration of an older layout), for no tracklets, and for an fps that is not a whole number, at least 1.
    """
    absent = [name for name in TRACKING_FIELDS if not hasattr(calibration, name)]
    if absent:
        raise ValueError("profile {!r} needs the Calibration's {}, and this one has no {}; pass calibration.load()"
                         .format(profile, " and ".join(TRACKING_FIELDS), " and no ".join(absent)))
    if _whole("fps", fps) < 1:
        raise ValueError("fps must be a whole number of at least 1, got {!r}".format(fps))
    seconds = np.asarray(calibration.tracklet_lengths_s, dtype=np.float64)
    if seconds.ndim != 1 or seconds.size == 0 or not np.isfinite(seconds).all():
        raise ValueError("the Calibration's tracklet_lengths_s must be a non-empty list of finite lengths in seconds, "
                         "its tracked spans; it has shape {}".format(seconds.shape))
    lengths = np.maximum(1, np.rint(seconds * fps)).astype(np.int64)
    return tracking_parameters(lengths, lost_share(calibration.present_counts))


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


def choose_tracklets(view, half, parameters, seed):
    """[Tracklet] of one half of a match, in order of (start, player), each with the lowest free slot of its pool.

    The walk (walk_run) is made on each player's runs of rows with a position, over the half's graded block, with
    rng_for(seed, view.match_index, "tracklets", half): players by column, runs in time order. Then the tracklets are
    taken in order of (start, player column) and each takes the lowest slot of its team's pool (home 0-23, away 24-45)
    that its previous tracklet has left: a slot is free from the stop of its last tracklet, and a tracklet that starts
    on that very row can take it. ValueError, naming the match, the half and the row, for a pool with no free slot;
    and, naming the half, for one the view has no block for.
    """
    start, stop = _block(view, half)
    present = ~np.isnan(np.asarray(view.players_xy)[start:stop, :, 0])
    team = np.asarray(view.player_team)
    rng = rng_for(seed, view.match_index, "tracklets", half)
    spans = []
    for player in range(present.shape[1]):
        firsts, stops = _runs(present[:, player])
        for run_start, run_stop in zip(firsts.tolist(), stops.tolist()):
            for first, last in walk_run(rng, run_stop - run_start, parameters):
                spans.append((start + run_start + first, player, start + run_start + last))
    spans.sort()
    free_from = [0] * N_SLOTS
    tracklets = []
    for first, player, last in spans:
        home = int(team[player]) == 0
        pool = range(HOME_SLOTS) if home else range(HOME_SLOTS, N_SLOTS)
        slot = next((candidate for candidate in pool if free_from[candidate] <= first), None)
        if slot is None:
            raise ValueError("{}: half {}: no free {} slot for the player in column {}, whose tracklet starts at frame "
                             "{}: the {} slots of its pool are all taken by players tracked at that frame. A team has "
                             "11 players on the pitch, so look at where the match's players_xy gives a position to a "
                             "player who is not playing".format(view.match_id, half, "home" if home else "away", player,
                                                                first, len(pool)))
        free_from[slot] = last
        tracklets.append(Tracklet(player=player, start=first, stop=last, slot=slot))
    return tracklets


def degraded_players(view, parameters, seed):
    """(players_xy, players_speed, tracklets): the match's players as slots, and the tracklets {half: [Tracklet]}.

    players_xy is (n, N_SLOTS, 2) and players_speed (n, N_SLOTS), NaN where no tracklet has the slot; a tracklet's slot
    holds the player's position on its rows, and its speed there is the step from the frame before in m/s on the
    match's pitch, NaN on the tracklet's first row. Rows outside the graded blocks have no players. Read-only.
    """
    source = np.asarray(view.players_xy, dtype=np.float64)
    pitch = np.asarray(view.pitch, dtype=np.float64)
    xy = np.full((view.n_frames, N_SLOTS, 2), np.nan)
    speed = np.full((view.n_frames, N_SLOTS), np.nan)
    records = {}
    for half, _, _ in view.graded:
        records[half] = choose_tracklets(view, half, parameters, seed)
        for t in records[half]:
            run = source[t.start:t.stop, t.player]
            xy[t.start:t.stop, t.slot] = run
            if t.stop - t.start > 1:
                step = np.diff(run * pitch, axis=0)
                speed[t.start + 1:t.stop, t.slot] = np.hypot(step[:, 0], step[:, 1]) * view.fps
    for array in (xy, speed):
        array.setflags(write=False)
    return xy, speed, records


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


def _players_record(view, parameters, records):
    """The JSON-ready account of the tracklets of every half: the parameters and, per half, counts. No frame in it."""
    halves, on_pitch, lost, count = {}, 0, 0, 0
    for half, start, stop in view.graded:
        frames = int((~np.isnan(np.asarray(view.players_xy)[start:stop, :, 0])).sum())
        tracked = sum(t.stop - t.start for t in records[half])
        halves[half] = {"tracklets": len(records[half]), "slots_used": len({t.slot for t in records[half]}),
                        "on_pitch_frames": frames, "lost_frames": frames - tracked}
        on_pitch, lost, count = on_pitch + frames, lost + frames - tracked, count + len(records[half])
    return {"d": round(parameters.d, DECIMALS), "mean_tracked_frames": round(parameters.mean_tracked, DECIMALS),
            "mean_lost_frames": round(parameters.mu, DECIMALS), "tracklets": count, "on_pitch_frames": on_pitch,
            "lost_frames": lost, "lost_share": round(lost / on_pitch, DECIMALS) if on_pitch else None,
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
    "players": the tracklets' parameters and counts (_players_record). ValueError for a profile that is not in
    PROFILES, for a seed that is not a whole number of at least 0, for a profile that needs the Calibration without
    one (or, for tracking, without the fields it reads: calibrated_parameters), and for a pool of slots with no free
    slot (choose_tracklets).
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
        xy, speed, records = degraded_players(view, parameters, seed)
        info["players"] = _players_record(view, parameters, records)
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
