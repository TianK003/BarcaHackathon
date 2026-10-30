"""Stage 1 -- inputs: every asset of the match, aligned to submission rows, loaded the first time it is asked for.

    inputs = Inputs()                 # touches nothing yet
    inputs.atd.ball_xy                # parses the ATD once (cached under data/cache/), then reuses it
    inputs.homography.matrices        # per-frame 3x3, checked to be one row per frame
    inputs.smart_tagging              # the 638 automatic phase intervals, start/end already in frames

Downstream of this module every per-frame array has exactly n_frames rows, and row i is frame i of the tactical
video -- the submission's own index. The conversions that make that true happen here and nowhere else:

    ATD             frame_id 1..199319         -> row frame_id - 1
    homography      frame 0..199318            -> row as is (checked)
    Smart Tagging   seconds from frame 0       -> round(seconds * 25), by starter/load_smart_tagging.py

pipeline/inputs/README.md explains the stage; docs/traps.md T08-T11 and T15-T19 are the traps it exists to absorb.
"""

import functools
import os
from dataclasses import dataclass, field

import numpy as np

import games
from pipeline.inputs import atd_cache
from pipeline.inputs.atd_cache import Atd  # noqa: F401  (re-exported: the type evidence code works with)
from starter import load_homography, load_smart_tagging

# Params
# Kick-off frames as the organisers document them (DATA.md, FAQ.md) and as starter/load_smart_tagging.py checks the
# tags against. The (minute, second, frame) anchors in games.py convert to 11186 and 113741 instead -- one frame off,
# in opposite directions (docs/traps.md T29). The documented pair is the one the graded windows agree with: the first
# half is graded from its kick-off, 11185.
KICKOFFS = {1: load_smart_tagging.KICKOFF_VIDEO_FRAMES["1st Half"],
            2: load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]}


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
    def camera_aim(self):
        """(n,) metres along the pitch under the tactical video's image centre (pipeline.inputs.camera), read-only.
        NaN where the homography is not valid. A property of the video, so a benchmark match has one only if given."""
        from pipeline.inputs import camera
        aim = camera.aim_x(self.homography.matrices, self.homography.valid, self.pitch[0])
        aim.flags.writeable = False
        return aim

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
