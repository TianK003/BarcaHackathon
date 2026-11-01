"""Stage 1 -- inputs: every asset of the match, aligned to submission rows, loaded the first time it is asked for.

    inputs = Inputs()                 # touches nothing yet
    inputs.atd.ball_xy                # parses the ATD once (cached under data/cache/), then reuses it
    inputs.homography.matrices        # per-frame 3x3, checked to be one row per frame
    inputs.smart_tagging              # the 638 automatic phase intervals, start/end already in frames
    inputs.with_ball_fill("broadcast").atd   # our ATD with the broadcast's ball in its gaps (R028, broadcast.py)

Downstream of this module every per-frame array has exactly n_frames rows, and row i is frame i of the tactical
video -- the submission's own index. The conversions that make that true happen here and nowhere else:

    ATD             frame_id 1..199319         -> row frame_id - 1
    homography      frame 0..199318            -> row as is (checked)
    Smart Tagging   seconds from frame 0       -> round(seconds * 25), by starter/load_smart_tagging.py
    broadcast       its own 164,488 frames     -> row t from its frame t - d, d found per half by code (R028)

The broadcast's own arrays (broadcast_atd, broadcast_valid) stay on its timeline; broadcast_ball is the one array of it
on our rows, and only a config with [inputs] ball_fill = "broadcast" reads any of it.

pipeline/inputs/README.md explains the stage; docs/traps.md T08-T11 and T15-T19 are the traps it exists to absorb.
"""

import copy
import functools
import os
from dataclasses import dataclass, field

import numpy as np

import games
from pipeline.inputs import atd_cache, broadcast
from pipeline.inputs.atd_cache import Atd  # noqa: F401  (re-exported: the type evidence code works with)
from starter import load_homography, load_smart_tagging

# Params
# Kick-off frames as the organisers document them (DATA.md, FAQ.md) and as starter/load_smart_tagging.py checks the
# tags against. The (minute, second, frame) anchors in games.py convert to 11186 and 113741 instead -- one frame off,
# in opposite directions (docs/traps.md T29). The documented pair is the one the graded windows agree with: the first
# half is graded from its kick-off, 11185.
KICKOFFS = {1: load_smart_tagging.KICKOFF_VIDEO_FRAMES["1st Half"],
            2: load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]}
BROADCAST_FOLDER = "atd_broadcast"     # the broadcast ATD's cache, beside the tactical's: <cache dir>/atd_broadcast/


@dataclass(frozen=True, eq=False)
class Homography:
    """The tactical homography, one row per frame. Maps NORMALISED pitch coordinates (0-1) to pixels.

    No array is in its repr (repr=False): numpy prints the head and tail of an array, and a Homography printed by a log
    line or a debugger would put frames of the match's homography in a log (the same as atd_cache.Atd).
    """
    matrices: np.ndarray = field(repr=False)  # (n, 3, 3) float64, NaN where valid is False (never, tactical)
    error: np.ndarray = field(repr=False)     # (n,) reprojection error -- lower is better, no upper bound
    valid: np.ndarray = field(repr=False)     # (n,) bool


class Inputs:
    """Every asset of one match, aligned to submission rows. Nothing is read until it is first used.

    The defaults describe the real match. The self-test overrides data_dir, n_frames and graded to run the same code
    on a small synthetic bundle; nothing else should need to.

    graded is the list of (half, start, stop) blocks the scorer grades -- half-open, from games.half_video_blocks:
    [(1, 11185, 90652), (2, 110808, 190198)]. The second half is graded from 2,934 frames before its kick-off, and
    all of those frames are D (docs/traps.md T02).
    """

    def __init__(self, game=games.GAME, data_dir=None, cache_dir=None, n_frames=None, graded=None, rebuild=False):
        self.game = game
        self.data_dir = data_dir or games.DATA_DIR
        self.cache_dir = cache_dir or os.path.join(self.data_dir, "cache")
        self.n_frames = n_frames or games.source_frames("tactical", game)
        self.fps = games.GAMES[game]["fps"]
        self.pitch = games.GAMES[game]["pitch"]  # metres, (length, width); the metadata's 105 x 68 is rounder
        self.graded = graded or [(half, first, last + 1) for half, first, last in games.half_video_blocks(game)]
        self.kickoffs = dict(KICKOFFS)
        self.rebuild = rebuild
        self.atd_status = "not loaded"  # then "built" (parsed this run) or "reused" (loaded from the cache)

    # -- tracking ------------------------------------------------------------------------------------------------------

    @functools.cached_property
    def _atd_and_manifest(self):
        meta_path, raw_path = games.require(["atd_meta", "atd_raw"], self.game, self.data_dir)
        atd, manifest, built = atd_cache.load_or_build(meta_path, raw_path, self.cache_dir, self.n_frames,
                                                       rebuild=self.rebuild)
        self.atd_status = "built" if built else "reused"
        return atd, manifest

    @property
    def atd(self):
        """The tracking as read-only arrays -- see atd_cache.Atd."""
        return self._atd_and_manifest[0]

    @property
    def atd_manifest(self):
        """The cache manifest: source hashes, versions, and aggregate statistics of the tracking."""
        return self._atd_and_manifest[1]

    def metres(self, xy):
        """Normalised (..., 2) coordinates to metres. Distances must be computed in metres: the pitch is not square,
        and skipping this stretches every distance along its short axis (docs/traps.md T16)."""
        return xy * np.array(self.pitch)

    # -- other assets --------------------------------------------------------------------------------------------------

    @functools.cached_property
    def homography(self):
        """The tactical homography, after checking it really is one row per frame, in order (docs/traps.md T08)."""
        matrices, error, valid = load_homography.load(self.game, self.data_dir)
        report = load_homography.check_alignment(self.game, self.data_dir)
        if not report["agrees"] or len(matrices) != self.n_frames:
            raise ValueError("the homography has {} rows for {} frames, or its frame column is not 0..n-1".format(
                len(matrices), self.n_frames))
        for array in (matrices, error, valid):
            array.flags.writeable = False
        return Homography(matrices=matrices, error=error, valid=valid)

    @functools.cached_property
    def smart_tagging(self):
        """The Smart Tagging intervals as starter/load_smart_tagging.load returns them: dicts with code, tags and
        start_frame/end_frame (half-open). Automatic phases, not possessions (docs/traps.md T17)."""
        return load_smart_tagging.load(self.game, self.data_dir)

    # -- the shape of the match ----------------------------------------------------------------------------------------

    @functools.cached_property
    def half_of_frame(self):
        """(n,) int8: 1 or 2 on graded frames, 0 elsewhere. Per-half features -- attacking direction swaps at
        halftime (T16) -- and per-half statistics start from this."""
        halves = np.zeros(self.n_frames, dtype=np.int8)
        for half, start, stop in self.graded:
            halves[start:stop] = half
        halves.flags.writeable = False
        return halves

    @property
    def graded_mask(self):
        """(n,) bool: frames the scorer grades. An X anywhere else costs nothing."""
        return self.half_of_frame > 0

    # -- the broadcast (R028) ------------------------------------------------------------------------------------------
    #
    # The broadcast is on its own timeline (docs/traps.md T09): broadcast_atd and broadcast_valid have one row per
    # BROADCAST frame, never one per ours. broadcast_ball is its one array on our rows, made by broadcast.py's steps.
    # Nothing here is read unless a config fills the ball (with_ball_fill("broadcast")).

    @functools.cached_property
    def _broadcast_atd_and_manifest(self):
        meta_path, raw_path = games.require(["atd_meta", "atd_raw"], self.game, self.data_dir, source="broadcast")
        atd, manifest, _ = atd_cache.load_or_build(meta_path, raw_path, self.cache_dir,
                                                   games.source_frames("broadcast", self.game), rebuild=self.rebuild,
                                                   folder=BROADCAST_FOLDER)
        return atd, manifest

    @property
    def broadcast_atd(self):
        """The broadcast's tracking as read-only arrays, one row per broadcast frame (164,488, 51 slots): an Atd parsed
        and cached by the tactical's own code (atd_cache), slot teams from its metadata as ours are."""
        return self._broadcast_atd_and_manifest[0]

    @property
    def broadcast_atd_manifest(self):
        """The broadcast ATD cache's manifest: its source files' sizes and hashes, versions, aggregate statistics."""
        return self._broadcast_atd_and_manifest[1]

    @functools.cached_property
    def _broadcast_homography(self):
        """(valid, the file's {"bytes", "sha256"}): of the broadcast homography only its valid column is used."""
        path = games.require(["homography"], self.game, self.data_dir, source="broadcast")[0]
        _, _, valid = load_homography.load(self.game, self.data_dir, source="broadcast")
        report = load_homography.check_alignment(self.game, self.data_dir, source="broadcast")
        if not report["agrees"]:
            raise ValueError("the broadcast homography has {} rows for {} broadcast frames, or its frame column is not "
                             "0..n-1".format(report["n_rows"], report["n_video_frames"]))
        valid.flags.writeable = False
        return valid, atd_cache.describe_source(path)

    @property
    def broadcast_valid(self):
        """(nb,) bool, read-only, one per broadcast frame: whether its homography exists. It does on 65.8% of them; the
        rest are replays, close-ups and crowd shots (A20)."""
        return self._broadcast_homography[0]

    @property
    def broadcast_homography_source(self):
        """{"bytes", "sha256"} of the broadcast homography file, for the run manifest."""
        return self._broadcast_homography[1]

    @functools.cached_property
    def broadcast_alignment(self):
        """broadcast.Alignment: the broadcast's offset in each graded half, found ball against ball in metres on every
        run (broadcast.align), or AlignmentError, which stops the run."""
        return broadcast.align(self.metres(self.atd.ball_xy), self.metres(self.broadcast_atd.ball_xy),
                               self.broadcast_valid, self.graded)

    @functools.cached_property
    def _broadcast_usable(self):
        """(ball, summary): the broadcast's usable ball on our rows, and the aggregates of how it was found."""
        ours, theirs, alignment = self.atd, self.broadcast_atd, self.broadcast_alignment
        fmap = broadcast.frame_map(alignment, self.graded, self.n_frames, theirs.n_frames)
        live = broadcast.live(self.metres(ours.players_xy), self.metres(theirs.players_xy), self.broadcast_valid, fmap)
        usable = broadcast.usable(self.metres(theirs.ball_xy), fmap, live)
        ball = np.full((self.n_frames, 2), np.nan)
        ball[usable] = theirs.ball_xy[fmap[usable]]
        ball.flags.writeable = False
        mapped = np.flatnonzero(fmap >= 0)
        valid_frames, live_frames = int(self.broadcast_valid[fmap[mapped]].sum()), int(live.sum())
        summary = {"offsets": dict(alignment.offsets), "pairs": dict(alignment.pairs),
                   "fine_scores": _rounded(alignment.fine_scores), "coarse_scores": _rounded(alignment.coarse_scores),
                   "runner_up": _rounded(alignment.runner_up), "valid_frames": valid_frames,
                   "live_frames": live_frames,
                   "live_share": round(live_frames / valid_frames, 6) if valid_frames else 0.0,
                   "usable_frames": int(usable.sum())}
        return ball, summary

    @property
    def broadcast_ball(self):
        """(n, 2) normalised, read-only: the broadcast's usable ball on our rows, on graded frames that are mapped and
        live and whose broadcast ball survives the 13-frame run filter, and NaN on every other row (broadcast.py)."""
        return self._broadcast_usable[0]

    @property
    def broadcast_summary(self):
        """Aggregates of how broadcast_ball was found, a new dict on each read: the alignment's offsets, pairs and
        scores per half; valid_frames, the graded frames mapped to a frame whose homography is valid; live_frames,
        those that are live; live_share, the second over the first; and usable_frames, those with a usable ball."""
        return copy.deepcopy(self._broadcast_usable[1])

    # -- the ball fill (R028) ------------------------------------------------------------------------------------------

    @functools.cached_property
    def ball_filled(self):
        """(n,) bool, read-only: the frames whose ball was filled. All False here: these inputs fill nothing, and
        with_ball_fill gives the inputs that do."""
        filled = np.zeros(self.n_frames, dtype=bool)
        filled.flags.writeable = False
        return filled

    @property
    def fill_summary(self):
        """The fill's aggregates, for the run's log and manifest; None here, where nothing is filled."""
        return None

    def with_ball_fill(self, kind):
        """These inputs with their ball filled as `kind` says ([inputs] ball_fill of a config, pipeline/config.py).

        "none" is the inputs object itself, so that every config without the fill runs exactly as it did. "broadcast"
        is a FilledInputs whose tracking has broadcast_ball wherever our ball is missing on a graded frame. Nothing is
        loaded here: the view reads the broadcast the first time its tracking is asked for.
        """
        if kind == "none":
            return self
        if kind == "broadcast":
            return FilledInputs(self)
        raise ValueError("with_ball_fill: kind {!r} is not one of: none, broadcast (a config's [inputs] ball_fill "
                         "names it)".format(kind))


def _rounded(scores):
    """{half: score} with the scores rounded to 6 decimals, as the cache manifest's shares are."""
    return {half: round(float(score), 6) for half, score in scores.items()}


def _from_base(name):
    """A property that reads `name` from the inputs a FilledInputs was made from: an asset the fill does not change."""
    return property(lambda self: getattr(self._base, name),
                    doc="The base inputs' {}: the fill does not change it.".format(name))


class FilledInputs(Inputs):
    """Inputs whose ball is filled from the broadcast where ours is missing (R028): Inputs.with_ball_fill("broadcast").

    It is an Inputs like any other, so every source reads `run.inputs.atd` and sees the filled tracking with no change
    of its own. Like the benchmark's MatchInputs it does not call Inputs.__init__: it copies the base's scalar fields
    (SCALARS), and reads from the base every asset the fill leaves as it is (the ATD's manifest and status, the
    homography, the Smart Tagging, every broadcast_*), so nothing is loaded twice. What it overrides:

        atd           the base's tracking with broadcast_ball in its gaps on graded frames (broadcast.fill). It is made
                      the first time it is read, and only then is the broadcast loaded, aligned and checked.
        ball_filled   (n,) bool, read-only: the frames the fill wrote.
        fill_summary  the base's broadcast_summary, with filled_frames and graded_frames: aggregates only.

    The per-object memos of the evidence (kinematics, the features, the models' scores) are its own, so a source never
    mixes the filled tracking with the plain one. It has the default object repr, so no array is ever in it.
    """

    SCALARS = ("game", "data_dir", "cache_dir", "n_frames", "fps", "pitch", "graded", "kickoffs", "rebuild")

    atd_status = _from_base("atd_status")
    atd_manifest = _from_base("atd_manifest")
    homography = _from_base("homography")
    smart_tagging = _from_base("smart_tagging")
    broadcast_atd = _from_base("broadcast_atd")
    broadcast_atd_manifest = _from_base("broadcast_atd_manifest")
    broadcast_valid = _from_base("broadcast_valid")
    broadcast_homography_source = _from_base("broadcast_homography_source")
    broadcast_alignment = _from_base("broadcast_alignment")
    broadcast_ball = _from_base("broadcast_ball")
    broadcast_summary = _from_base("broadcast_summary")

    def __init__(self, base):
        self._base = base
        for name in self.SCALARS:
            setattr(self, name, copy.copy(getattr(base, name)))

    @functools.cached_property
    def _fill(self):
        return broadcast.fill(self._base.atd, self._base.broadcast_ball, self.graded_mask)

    @property
    def atd(self):
        """The base's tracking with the broadcast's usable ball in its gaps, as read-only arrays (broadcast.fill)."""
        return self._fill[0]

    @property
    def ball_filled(self):
        """(n,) bool, read-only: the frames whose ball the fill wrote (graded, ours missing, the broadcast's usable)."""
        return self._fill[1]

    @property
    def fill_summary(self):
        """The base's broadcast_summary with filled_frames and graded_frames, a new dict on each read. Aggregates
        only: it goes into the run's log line and its manifest."""
        summary = dict(self._base.broadcast_summary)
        summary.update(filled_frames=int(self.ball_filled.sum()), graded_frames=int(self.graded_mask.sum()))
        return summary
