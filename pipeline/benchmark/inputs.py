"""A DFL match served to the pipeline as if it were ours: MatchInputs, so that a config runs on it unchanged.

    match  = dfl.load_match("J03WMX")                         # tracking and the ball's own flags, from the cache
    inputs = MatchInputs(match, clean_atd(match))             # what a component sees as `run.inputs`
    codes, sources = run.predict(config, inputs)              # pipeline.run's own functions, on this match

Why it is a subclass of pipeline.inputs.Inputs. Every evidence source, decoder and refiner of the pipeline is written
against `run.inputs`: its `atd` (the tracking), `pitch` and `metres`, `n_frames`, `graded` and `half_of_frame`. A
benchmark that scored something else than those components would measure nothing about them, so a benchmark match is
served through the same class and the same fields, and a component cannot tell it from BAR-ATH. Inputs.__init__ is not
called: it would load BAR-ATH's constants (its frame count, its graded halves, its kick-offs) and it names files that a
DFL match does not have. The fields it sets are set here from the match instead, and half_of_frame and graded_mask,
which read only n_frames and graded, are the base class's own.

What a benchmark match does not have, it says. The homography and the Smart Tagging come from the video, and the DFL
open data has no video: asking for either raises AssetUnavailable, with a sentence that says to check that source on
the public window instead. Nothing is faked, so a component that needs them fails loudly instead of learning from zeros.

No truth through the inputs. Evidence components see `ctx.run.inputs`, and the DFL match carries the answer in its own
ball_status and ball_possession (truth.py builds the benchmark's truth from them). A MatchInputs that kept the match,
or any array made of those two flags, would let a component read the answer, and the benchmark would quietly measure
that leak. So only the scalars are copied out of the match (its id, length, rate, pitch, halves and kick-offs) and the
tracking is held as an Atd, which has positions and speeds and no flags. The checks assert it
(checks/inputs.py: test_match_inputs_hold_no_truth walks everything the inputs reach, to any depth, for the match, the
flags and the codes made of them). A second check asserts that no component under pipeline/evidence, decode or refine
imports pipeline.benchmark, which would let it load the truth itself by the match id. Whatever oracle a check wants
gets the truth by closure from its own test, never from here.

clean_atd is the tracking as DFL gives it. The degrader (a later module) makes an Atd of the same layout from it, with
BAR-ATH's gaps in the ball, and serves that through MatchInputs with its own profile and seed, which name where the
tracking came from in the manifest and in `cache_dir`. Nothing is written there: the inputs are held in memory, and
cache_dir only says where the pipeline would look for anything it cached for this (match, profile, seed).
"""

import json
import os

import numpy as np

import games
from pipeline.benchmark import dfl
from pipeline.inputs import Inputs
from pipeline.inputs.atd_cache import Atd

# Params
ASSET_MESSAGE = ("{0}: the benchmark matches have no {0} -- the DFL open data comes without video or tagging (P01). "
                 "Check this source on the public window instead.")
CACHE_FOLDER = ("cache", "benchmark", "inputs")     # <data dir>/cache/benchmark/inputs/<match>/<profile>/<seed>
ATD_DTYPES = {"players_xy": np.float64, "players_speed": np.float64, "ball_xy": np.float64,
              "ball_speed": np.float64}             # slot_team is int8: the BAR-ATH cache's own dtypes (atd_cache.Atd)


class AssetUnavailable(LookupError):
    """The asset a component asked of the inputs does not exist for a benchmark match (it has no video)."""


def clean_atd(match):
    """The tracking of a DflMatch as an Atd, untouched: DFL's positions, speeds and ball, its players as the slots.

    The same fields, dtypes and dimensions as the Atd that atd_cache builds from BAR-ATH's raw file, with slot_team
    the players' teams (0 home, 1 away, as DflMatch.player_team has them: a DFL "slot" is a squad member, of whom
    only those on the pitch have positions, NaN elsewhere). Every array is read-only. An array of the match that is
    already of the right dtype and read-only (which is what dfl.load_match gives) is used as it is, with no copy; one
    that is writeable, or of another dtype, becomes a read-only view or a cast copy, and the match's own array is never
    frozen or changed by this.

    ValueError, naming the match, where a position has one coordinate and not the other: the Atd's invariant is that x
    and y are missing together (atd_cache._check_presence on BAR-ATH's file), and nearest_player tells a present slot
    by its x alone. The 7 downloaded matches have no such position; this is for a hand-built one.
    """
    arrays = {name: _frozen(getattr(match, name), dtype) for name, dtype in ATD_DTYPES.items()}
    atd = Atd(slot_team=_frozen(match.player_team, np.int8), **arrays)
    slots = int((np.isnan(atd.players_xy[:, :, 0]) != np.isnan(atd.players_xy[:, :, 1])).sum())
    balls = int((np.isnan(atd.ball_xy[:, 0]) != np.isnan(atd.ball_xy[:, 1])).sum())
    if slots or balls:
        raise ValueError("{}: x and y are missing apart on {} slot-frames and {} ball frames; the pipeline reads a "
                         "position as present when its x is, so decide what a half-missing position means before "
                         "serving this match".format(match.match_id, slots, balls))
    return atd


def _frozen(array, dtype):
    array = np.asarray(array, dtype=dtype)           # the same object when it already has the dtype
    if array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


class MatchInputs(Inputs):
    """The inputs of one benchmark match: `match`'s scalars and `atd`'s tracking, in the shape the pipeline reads.

    `match` is a DflMatch and `atd` the tracking to serve for it (clean_atd(match), or a degraded one of the same
    length); `profile` and `seed` name how that tracking was made, and go into `cache_dir` and the manifest. `data_dir`
    defaults to games.DATA_DIR. The match is read here and not kept (see the module docstring): only its id, length,
    rate, pitch, halves and kick-offs are, and the sources recorded by its cache, for the manifest.

    The fields are Inputs' own: game (the match id), data_dir, cache_dir, n_frames, fps, pitch (the match's own, in
    metres, so `metres` gives DFL distances), graded (a list of (half, start, stop), half-open), kickoffs (the first
    row of each half), rebuild (False) and atd_status ("benchmark"). `atd` and `atd_manifest` are properties.

    ValueError if the Atd and the match differ in length: a row of the tracking would then be another frame than the
    match's, and every score computed from it would be quietly wrong.
    """

    def __init__(self, match, atd, profile="clean", seed=0, data_dir=None):
        if atd.n_frames != match.n_frames:
            raise ValueError("{}: the Atd has {} rows and the match has {}; the tracking served must be the match's "
                             "own (clean_atd(match), or a degraded copy of it)".format(
                                 match.match_id, atd.n_frames, match.n_frames))
        self.game = match.match_id
        self.data_dir = data_dir or games.DATA_DIR
        self.profile = profile
        self.seed = seed
        self.cache_dir = os.path.join(self.data_dir, *CACHE_FOLDER, match.match_id, profile, str(seed))
        self.n_frames = match.n_frames
        self.fps = match.fps
        self.pitch = tuple(match.pitch)
        self.graded = [tuple(block) for block in match.graded]
        self.kickoffs = dict(match.kickoffs)
        self.rebuild = False
        self.atd_status = "benchmark"
        self._atd = atd
        cache, reason = _dfl_cache_manifest(match, self.data_dir)
        self._atd_manifest = {"match_id": match.match_id, "profile": profile, "seed": seed,
                              "cache_version": cache.get("cache_version"), "sources": cache.get("sources"),
                              "sources_reason": reason}

    @property
    def atd(self):
        """The tracking served, as read-only arrays: clean_atd(match) or a degraded copy of it."""
        return self._atd

    @property
    def atd_manifest(self):
        """What a report says about the run's inputs: the match, the profile and seed, and the DFL source files.

        `sources` is {"information": {"bytes", "sha256"}, "positions": {...}} as the match's DFL cache recorded them,
        so that a result names the files it came from, and `sources_reason` is None. It is copied only from a cache
        that is of this very match: the id alone finds the cache, so the manifest's match_id, fps, pitch and graded
        must also be the match's own. Otherwise (no cache, one that cannot be read, one of another match) `sources` and
        `cache_version` are None and `sources_reason` says why.
        """
        return self._atd_manifest

    @property
    def homography(self):
        raise AssetUnavailable(ASSET_MESSAGE.format("homography"))

    @property
    def smart_tagging(self):
        raise AssetUnavailable(ASSET_MESSAGE.format("smart_tagging"))


def _dfl_cache_manifest(match, data_dir):
    """(its manifest, None) if the match's DFL cache under `data_dir` is of this match, else ({}, the reason why not).

    "Of this match" is that the manifest's match_id, fps, pitch and graded are the match's own, and that it records its
    sources: a hand-built match under a real id, or a cache folder copied under another id, is not served as made from
    the files of the match whose id it carries.
    """
    path = os.path.join(data_dir, *dfl.CACHE_FOLDER, match.match_id, dfl.MANIFEST)
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except OSError:
        return {}, ("no DFL cache manifest for {} (the match was not read through dfl.load_match, or its cache was "
                    "removed)".format(match.match_id))
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict):
        return {}, "the DFL cache manifest of {} cannot be read as a JSON object".format(match.match_id)
    own = {"match_id": match.match_id, "fps": match.fps, "pitch": [float(side) for side in match.pitch],
           "graded": [[int(value) for value in block] for block in match.graded]}
    differ = ["{} is {} and the manifest says {}".format(name, value, manifest.get(name))
              for name, value in own.items() if manifest.get(name) != value]
    if differ:
        return {}, "the DFL cache manifest filed under {} is of another match: {}".format(
            match.match_id, "; ".join(differ))
    if not isinstance(manifest.get("sources"), dict):
        return {}, "the DFL cache manifest of {} records no sources".format(match.match_id)
    return manifest, None
