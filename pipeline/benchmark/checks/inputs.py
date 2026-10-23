"""Checks of the benchmark's inputs (pipeline/benchmark/inputs.py): a DFL match served to the pipeline as if ours.

The synthetic checks run on a DflMatch built here out of random numbers, in the real layout (one timeline, the halves
a gap apart, NaN positions and codes of -1 on the gap), so that every answer can be worked out without a download. The
question each one asks is whether a component written for BAR-ATH, which sees only `inputs`, gets from a benchmark match
what it gets from ours: the same fields, the same units, the same read-only arrays, and nothing it could not get from
ours, which is the truth. The one that reads a downloaded match is test_dfl_* (--dfl); the DFL files are public
(CC BY 4.0), so printing its aggregates is fine.
"""

import dataclasses
import json
import os
import tempfile

import numpy as np

import games
from pipeline.benchmark import dfl, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark.checks.common import real_match, write_synthetic_dfl
from pipeline.inputs import atd_cache
from pipeline.selftest import expect_error, make_synthetic_bundle, synthetic_inputs

# Params
HOME, AWAY = 0, 1
ASSET_MESSAGE = ("{0}: the benchmark matches have no {0} -- the DFL open data comes without video or tagging (P01). "
                 "Check this source on the public window instead.")
# Rows where nearest_player and truth.nearest_team name different teams, per match: exact ties between the best home and
# the best away player, which the two computations break differently. TIE_DISTANCE (m2) is how close the two best
# distances must be for a row to count as one: the one tie there is differs by 5e-15.
EXPECTED_TIES = {"J03WMX": (58523,)}
TIE_DISTANCE = 1e-9
# What an Atd is made of, as the BAR-ATH cache builds it: (dtype, dimensions) of each array.
ATD_LAYOUT = {"players_xy": (np.float64, 3), "players_speed": (np.float64, 2), "ball_xy": (np.float64, 2),
              "ball_speed": (np.float64, 1), "slot_team": (np.int8, 1)}


# -- helpers ----------------------------------------------------------------------------------------------------------

def random_match(match_id="RND001", seed=0, rows=(240, 200), gap=25, teams=(0, 0, 0, 1, 1, 1), pitch=(105.0, 68.0),
                 fps=25, writeable=False):
    """A DflMatch of random positions and flags: the halves are rows[0] and rows[1] long, `gap` rows apart.

    Positions wander a little outside the pitch and about a quarter of the players, and a fifth of the balls, are
    missing, as they are in the real data; nothing is a tie. The gap rows are what DFL's separator is: no ball, no
    players, codes -1. The arrays are read-only, as dfl.load_match gives them, unless `writeable`.
    """
    rng = np.random.default_rng([9, seed])
    n, players = sum(rows) + gap, len(teams)
    graded = ((1, 0, rows[0]), (2, rows[0] + gap, n))
    inside = np.zeros(n, dtype=bool)
    for _, start, stop in graded:
        inside[start:stop] = True

    xy = rng.uniform(-0.02, 1.02, size=(n, players, 2))
    xy[rng.uniform(size=(n, players)) < 0.25] = np.nan
    ball = rng.uniform(-0.02, 1.02, size=(n, 2))
    ball[rng.uniform(size=n) < 0.2] = np.nan
    arrays = {
        "frame_number": np.where(inside, np.arange(n), -1).astype(np.int64),
        "players_xy": xy, "players_speed": rng.uniform(0, 9, size=(n, players)),
        "player_team": np.array(teams, dtype=np.int8), "ball_xy": ball, "ball_z": rng.uniform(0, 3, size=n),
        "ball_speed": rng.uniform(0, 30, size=n),
        "ball_status": rng.integers(0, 2, size=n).astype(np.int8),
        "ball_possession": rng.integers(0, 2, size=n).astype(np.int8)}
    for name in ("players_xy", "players_speed", "ball_xy", "ball_z", "ball_speed"):
        arrays[name][~inside] = np.nan
    for name in ("ball_status", "ball_possession"):
        arrays[name][~inside] = -1
    for array in arrays.values():
        array.setflags(write=writeable)
    return dfl.DflMatch(match_id=match_id, pitch=pitch, fps=fps, graded=graded,
                        kickoffs={half: start for half, start, _ in graded}, report={}, **arrays)


def truth_forms(match):
    """Every array a component could rebuild the answer from, or is the answer: the flags and the codes made of them."""
    status, possession = np.asarray(match.ball_status), np.asarray(match.ball_possession)
    codes, _ = truth.truth_codes(match)
    return {"ball_status": status, "ball_possession": possession, "alive": status == 1, "dead": status == 0,
            "home": possession == HOME, "away": possession == AWAY, "codes": codes}


def leaks(inputs, match):
    """The attributes of `inputs`, and of its Atd, that are a DflMatch or an array equal to the flags or the truth."""
    forms = truth_forms(match)
    found = []
    held = dict(vars(inputs))
    held.update({"atd." + name: value for name, value in vars(inputs.atd).items()})
    for name, value in held.items():
        if isinstance(value, dfl.DflMatch):
            found.append(name)
        elif isinstance(value, np.ndarray) and value.dtype != object:
            found += ["{} == {}".format(name, form) for form, array in forms.items()
                      if value.shape == array.shape and np.array_equal(value, array)]
    return found


# -- the contract -----------------------------------------------------------------------------------------------------

def test_match_inputs_contract():
    """MatchInputs is an Inputs for a benchmark match: the same fields the pipeline's components read, and no truth."""
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    from pipeline.inputs import Inputs
    match = random_match()
    atd = match_inputs.clean_atd(match)
    with tempfile.TemporaryDirectory() as folder:
        inputs = match_inputs.MatchInputs(match, atd, data_dir=folder)
        assert isinstance(inputs, Inputs)
        assert inputs.game == "RND001" and inputs.data_dir == folder and inputs.rebuild is False
        assert inputs.cache_dir == os.path.join(folder, "cache", "benchmark", "inputs", "RND001", "clean", "0"), \
            inputs.cache_dir
        assert inputs.n_frames == 465 and inputs.fps == 25 and inputs.pitch == (105.0, 68.0)
        assert inputs.atd_status == "benchmark"
        assert isinstance(inputs.graded, list) and inputs.graded == [(1, 0, 240), (2, 265, 465)], inputs.graded
        assert inputs.kickoffs == {1: 0, 2: 265} == match.kickoffs
        assert inputs.atd is atd and inputs.atd_manifest["profile"] == "clean" and inputs.atd_manifest["seed"] == 0
        assert os.listdir(folder) == [], "nothing is written: the inputs are served from memory"

    default = match_inputs.MatchInputs(match, atd)
    assert default.data_dir == games.DATA_DIR and (default.profile, default.seed) == ("clean", 0)
    assert default.cache_dir == os.path.join(games.DATA_DIR, "cache", "benchmark", "inputs", "RND001", "clean", "0")
    other = match_inputs.MatchInputs(match, atd, profile="degraded", seed=4)
    assert other.cache_dir.endswith(os.path.join("RND001", "degraded", "4")), other.cache_dir

    # The shape of the match, from the base class: 1 or 2 inside a half, 0 on the gap between them.
    halves = inputs.half_of_frame
    assert halves.dtype == np.int8 and halves.shape == (465,) and not halves.flags.writeable
    assert halves[0] == 1 and halves[239] == 1 and (halves[240:265] == 0).all() and halves[265] == 2
    assert halves[464] == 2
    assert inputs.graded_mask.sum() == 440 and not inputs.graded_mask[240:265].any()
    assert inputs.graded_mask[:240].all() and inputs.graded_mask[265:].all()
    # Metres are the match's own pitch, 105 x 68 here, so a distance in them is a DFL distance.
    assert np.array_equal(inputs.metres(np.array([[0.5, 0.5], [1.0, 0.25]])), np.array([[52.5, 34.0], [105.0, 17.0]]))
    wide = random_match(pitch=(100.0, 50.0))
    square = match_inputs.MatchInputs(wide, match_inputs.clean_atd(wide))
    assert np.array_equal(square.metres(np.array([[1.0, 1.0]])), np.array([[100.0, 50.0]]))

    # The pipeline's own nearest_player, on these inputs, names the team that truth.nearest_team does: home = 1.0 where
    # the nearest player is team 0, 0.0 where it is team 1, NaN (no opinion) where there is no ball or nobody.
    evidence = REGISTRY["nearest_player"](RunContext(inputs=inputs, seed=0))
    nearest = truth.nearest_team(match)
    expected = np.where(nearest == HOME, 1.0, np.where(nearest == AWAY, 0.0, np.nan))
    assert {int(team) for team in np.unique(nearest)} == {-1, HOME, AWAY}, "the match must reach every branch"
    differ = ~((evidence.home == expected) | (np.isnan(evidence.home) & np.isnan(expected)))
    assert not differ.any(), "{} frames differ, first at row {}".format(
        int(differ.sum()), int(np.flatnonzero(differ)[0]))
    assert np.isnan(evidence.alive).all(), "nearest_player has no opinion on alive vs dead"

    # What a BAR-ATH asset is and a benchmark match has none of.
    for asset in ("homography", "smart_tagging"):
        message = expect_error(match_inputs.AssetUnavailable, lambda: getattr(inputs, asset), asset, "P01")
        assert message == ASSET_MESSAGE.format(asset), message
    assert issubclass(match_inputs.AssetUnavailable, LookupError)

    # Read-only, like the BAR-ATH Atd and for the same reason: one component must not change what the next one reads.
    for name in ATD_LAYOUT:
        assert not getattr(inputs.atd, name).flags.writeable, name
    assert not inputs.atd.players_present.flags.writeable and not inputs.atd.ball_present.flags.writeable

    # And no truth: nothing it holds is the match, the flags, or a mask or code made of them (test_match_inputs_hold_no_
    # truth tests the detector itself).
    assert leaks(inputs, match) == [], leaks(inputs, match)


def test_match_inputs_hold_no_truth():
    """Evidence components see ctx.run.inputs, so the match's flags must not be reachable through it.

    Not the DflMatch, not the flags, not an alive/dead or home/away mask made of them, not the truth's codes: neither
    as an attribute of the inputs nor as a field of its Atd, even after the lazy properties have been read.
    """
    match = random_match()
    inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
    inputs.half_of_frame, inputs.graded_mask, inputs.atd.players_present, inputs.atd.ball_present
    assert leaks(inputs, match) == [], leaks(inputs, match)

    # The detector detects: a held match, a held flag array, a held mask and a held truth are each found.
    for name, value in (("match", match), ("status", np.array(match.ball_status)),
                        ("mask", match.ball_status == 1), ("codes", truth.truth_codes(match)[0]),
                        ("possession", np.array(match.ball_possession).astype(np.int16))):
        probe = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
        setattr(probe, name, value)
        assert leaks(probe, match), "{}: a held {} was not found".format(name, name)

    # Everything it does keep is a scalar of the match or the tracking.
    kept = set(vars(match_inputs.MatchInputs(match, match_inputs.clean_atd(match))))
    held_back = {name for name in kept if "ball" in name or "possession" in name or name in ("match", "truth")}
    assert not held_back, sorted(kept)


# -- clean_atd --------------------------------------------------------------------------------------------------------

def test_clean_atd_has_the_layout_of_the_bar_ath_cache():
    """The same fields, dtypes and dimensions as the Atd that atd_cache builds, with slot_team the players' teams."""
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        real = synthetic_inputs(root).atd
    assert {name: (getattr(real, name).dtype.type, getattr(real, name).ndim) for name in atd_cache.ARRAYS} == ATD_LAYOUT

    match = random_match()
    atd = match_inputs.clean_atd(match)
    assert isinstance(atd, atd_cache.Atd) and atd_cache.ARRAYS == tuple(ATD_LAYOUT)
    for name, (dtype, ndim) in ATD_LAYOUT.items():
        array = getattr(atd, name)
        assert array.dtype == dtype and array.ndim == ndim and not array.flags.writeable, name
    assert (atd.n_frames, atd.n_slots) == (match.n_frames, match.n_players) == (465, 6)
    assert atd.players_xy.shape == (465, 6, 2) and atd.players_speed.shape == (465, 6)
    assert atd.slot_team.tolist() == match.player_team.tolist() == [0, 0, 0, 1, 1, 1]
    # What the match holds is not copied: the arrays are the match's own, already read-only.
    for name in ("players_xy", "players_speed", "ball_xy", "ball_speed"):
        assert getattr(atd, name) is getattr(match, name), name
    assert atd.slot_team is match.player_team
    assert np.array_equal(atd.ball_present, ~np.isnan(match.ball_xy[:, 0]))


def test_clean_atd_never_changes_the_match_and_casts_what_does_not_fit():
    loose = random_match(writeable=True)
    atd = match_inputs.clean_atd(loose)
    for name in atd_cache.ARRAYS:
        assert not getattr(atd, name).flags.writeable, "{} must be read-only".format(name)
    for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "player_team"):
        assert getattr(loose, name).flags.writeable, "{}: the match's own array was frozen".format(name)
    assert np.shares_memory(atd.players_xy, loose.players_xy), "a view, not a copy"

    # A match whose arrays are of other dtypes (a hand-built one) is cast to the Atd's.
    base = random_match()
    odd = dataclasses.replace(base, players_xy=np.array(base.players_xy, dtype=np.float32),
                              player_team=np.array(base.player_team, dtype=np.int64))
    cast = match_inputs.clean_atd(odd)
    assert cast.players_xy.dtype == np.float64 and cast.slot_team.dtype == np.int8
    assert cast.slot_team.tolist() == [0, 0, 0, 1, 1, 1]
    assert np.array_equal(cast.players_xy, odd.players_xy.astype(np.float64), equal_nan=True)
    assert all(not getattr(cast, name).flags.writeable for name in atd_cache.ARRAYS)


def test_match_inputs_refuses_an_atd_of_another_length_and_names_what_was_run():
    match = random_match()
    short = random_match(rows=(100, 100))
    expect_error(ValueError, lambda: match_inputs.MatchInputs(match, match_inputs.clean_atd(short)), "RND001", "225",
                 "465")

    # The manifest is what a report says about the run: the match's source files, as the DFL cache recorded them, and
    # the profile and the seed of the tracking served.
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "source")
        os.makedirs(source)
        real = dfl.load_match("SYN001", data_dir=root, paths=write_synthetic_dfl(source))
        with open(os.path.join(root, "cache", "dfl", "SYN001", "manifest.json"), encoding="utf-8") as handle:
            cache = json.load(handle)
        inputs = match_inputs.MatchInputs(real, match_inputs.clean_atd(real), profile="degraded", seed=3, data_dir=root)
        manifest = inputs.atd_manifest
        assert manifest["sources"] == cache["sources"] and sorted(manifest["sources"]) == ["information", "positions"]
        assert all(len(source["sha256"]) == 64 for source in manifest["sources"].values())
        assert (manifest["match_id"], manifest["profile"], manifest["seed"]) == ("SYN001", "degraded", 3), manifest
        assert manifest["cache_version"] == cache["cache_version"]
        json.dumps(manifest, sort_keys=True)                         # it goes into a report: JSON-ready
        # A match that was not read through the cache has no sources to name, and says so.
        bare = match_inputs.MatchInputs(match, match_inputs.clean_atd(match), data_dir=root)
        assert bare.atd_manifest["sources"] is None


# -- a real match -----------------------------------------------------------------------------------------------------

def test_dfl_match_inputs_serve_the_real_matches():
    """The 7 downloaded matches through MatchInputs: the fields, the read-only arrays, no truth, the nearest player.

    The pipeline's nearest_player on a DFL match must name the team that truth.nearest_team does on every frame, which
    is the home mapping (home = 1.0 where team 0 is nearest) proven on real data. The two compute the distance with
    different floating-point operations, so a frame where the best home player and the best away player are exactly
    as far from the ball can be broken either way. One such frame exists in the 7 matches (EXPECTED_TIES: J03WMX row
    58523, where two players, one of each team, are at the same x and 0.34 m either side of the ball across the pitch,
    their distances equal to 5e-15 m2); the check says so and requires that every row where the two differ is such a
    tie, and that there are no others. The comparison is not loosened for any other frame.
    """
    from pipeline.benchmark import sources
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    print("      {:<8} {:>9} {:>11} {:>11}".format("match", "rows", "answered", "tie rows"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
        assert inputs.game == match_id and inputs.n_frames == match.n_frames and inputs.fps == 25
        assert inputs.graded == [tuple(block) for block in match.graded] and inputs.kickoffs == match.kickoffs
        assert inputs.pitch == match.pitch
        assert inputs.graded_mask.sum() == sum(stop - start for _, start, stop in match.graded)
        assert all(not getattr(inputs.atd, name).flags.writeable for name in atd_cache.ARRAYS)
        assert all(getattr(inputs.atd, name) is getattr(match, field) for name, field in (
            ("players_xy", "players_xy"), ("ball_xy", "ball_xy"), ("slot_team", "player_team"))), "copied, not served"
        assert leaks(inputs, match) == [], leaks(inputs, match)
        assert all(len(source["sha256"]) == 64 for source in inputs.atd_manifest["sources"].values())

        evidence = REGISTRY["nearest_player"](RunContext(inputs=inputs, seed=0))
        nearest = truth.nearest_team(match)
        expected = np.where(nearest == HOME, 1.0, np.where(nearest == AWAY, 0.0, np.nan))
        differ = ~((evidence.home == expected) | (np.isnan(evidence.home) & np.isnan(expected)))
        rows = tuple(np.flatnonzero(differ).tolist())
        scale = np.array(match.pitch)
        for row in rows:
            distance = (((match.players_xy[row] - match.ball_xy[row]) * scale) ** 2).sum(axis=1)
            distance = np.where(np.isnan(distance), np.inf, distance)
            best = [float(distance[match.player_team == team].min()) for team in (HOME, AWAY)]
            assert abs(best[0] - best[1]) < TIE_DISTANCE, (
                "{}: row {} differs and is not a tie: the best home and away distances are {} and {} m2".format(
                    match_id, row, *best))
        print("      {:<8} {:>9,} {:>11.1%} {:>11}".format(match_id, match.n_frames, (~np.isnan(evidence.home)).mean(),
                                                         len(rows)))
        assert rows == EXPECTED_TIES.get(match_id, ()), "{}: the tied rows are {}, expected {}".format(
            match_id, rows, EXPECTED_TIES.get(match_id, ()))
