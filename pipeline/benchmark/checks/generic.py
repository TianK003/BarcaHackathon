"""Checks of the `generic` profile of the degrader (pipeline/benchmark/degrade.py), P27: the benchmark's default, and
the only degraded profile that reads nothing of our match.

What is held against it here is what its design declares (docs/superpowers/specs/2026-11-09-p27-degradation-design.md),
part by part:

* the constants are the declared ones, each a figure that the organisers publish in DATA.md or a rule declared before
  any run, and the arithmetic that the design quotes follows from them (test_generic_constants_are_the_declared_ones);
* no Calibration: one that is given is refused, none is loaded by the degrader, the runner, the trainers or the
  instruments, and the three functions of the profile name nothing of the Calibration
  (test_generic_takes_no_calibration_and_refuses_one, test_generic_no_instrument_loads_a_calibration); and no gate
  against our match: the fidelity report refuses it (test_generic_is_held_to_no_fidelity_gate_against_our_match);
* no truth: the view has no flag, and two matches that differ only in their flags get the same bytes
  (test_generic_never_sees_the_truth);
* the draws: the same seed gives the same bytes whatever other matches are run (test_generic_is_deterministic_and_
  independent_of_the_other_matches), and the ball's mask is the documented sequence of draws, restated here one frame at
  a time from the documentation with the declared numbers written out, not read from the code
  (reference_mask, test_generic_mask_is_the_documented_draws);
* the ball is missing exactly where each half's mask says, from the kick-off, and untouched elsewhere; its draws are
  the "generic_gaps" stream's own and look at neither the play nor the players
  (test_generic_ball_is_missing_exactly_where_the_masks_say);
* the players are the tracking profile's machinery with the declared parameters and none of ours: lost at the declared
  rate, in 46 slots that never change player between two rows (test_generic_parameters_are_the_declared_rules,
  test_generic_players_are_lost_at_the_declared_rate_in_46_slots);
* the runner serves it per seed with no Calibration in the report, and files it under its own name
  (test_generic_runs_on_the_runner_with_no_calibration).

Everything runs on the synthetic matches of checks/common.py: none reads data, and none of the numbers is a figure of
any match. The checks of what every profile shares (rng_for, the view, the transplant) are checks/degrade.py's, those
of the players' machinery (the walk, the slots, the speeds, the smoothing) checks/tracking.py's, and the runner's own
checks/run.py's, which also holds the default profile's run and its bytes across processes.
"""

import ast
import contextlib
import dataclasses
import inspect
import io
import json
import math
import os
import tempfile

import numpy as np

from pipeline.benchmark import calibration, conventions, degrade, diagnose, fidelity, restarts, score, sources
from pipeline.benchmark import touch_stats, train_dead_ball, train_shape, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (leaks, loading, made_up_calibration, random_match, read, smoothed_plain,
                                              steady_match, strict_json, synthetic_calibration, synthetic_matches,
                                              tracking_calibration, yardstick_path)
from pipeline.config import load_config
from pipeline.selftest import expect_error, quiet, write_text

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPEC = "docs/superpowers/specs/2026-11-09-p27-degradation-design.md"
# What every message about a constant says: where it is declared, where it comes from, and that it is not ours to tune.
DECLARED = ("the generic profile's constants are declared in {}, before any run: each is a figure that the organisers "
            "publish in DATA.md or a declared rule, never a measurement of our match, and none may be changed because "
            "of a score, a gate or anything seen on our match".format(SPEC))
LONG_ROWS = (3000, 2600)            # halves long enough for several gaps of the ball and several lost spans of a player
SQUAD = (0,) * 11 + (1,) * 11       # 22 players, home first: a full-strength match
SQUAD_ROWS = (30000, 30000)         # 20 minutes a half at 25 Hz: about 19 tracked and lost spans of every player
INFO_KEYS = ["ball", "mask_share", "match_id", "match_index", "players", "profile", "seed", "version"]
ARRAYS = ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team")
# A config with the hmm decoder, which diagnose needs (a majority vote has no posterior).
HMM_CONFIG = """
scope = "halves"

[[evidence]]
kind = "nearest_player"

[[evidence]]
kind = "ball_presence"

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 8
min_possession = 3
min_dead = 3
"""
TOUCH_SETTING = {"radius": 40.0, "max_rel_speed": math.inf, "min_frames": 2}


# -- helpers ----------------------------------------------------------------------------------------------------------

def everything(atd):
    """The bytes of every array of a degraded Atd: equal bytes are the same tracking, to the last bit."""
    return b"".join(getattr(atd, name).tobytes() for name in ARRAYS)


def players_bytes(atd):
    return atd.players_xy.tobytes() + atd.players_speed.tobytes() + atd.slot_team.tobytes()


def runs_of(flags):
    """[(first, stop)] of the maximal runs of True in a list of flags, written out one frame at a time."""
    runs, first = [], None
    for index, flag in enumerate(flags):
        if flag and first is None:
            first = index
        elif not flag and first is not None:
            runs.append((first, index))
            first = None
    if first is not None:
        runs.append((first, len(flags)))
    return runs


def reference_mask(rng, n, fps):
    """The generic profile's missing-ball mask of n frames at `fps` as its documentation states it: a list of bools.

    The declared numbers are written out here (41% missing; gaps log-uniform between 1 s and 60 s), not read from the
    code. The draws, in order: one uniform for the first state (missing if it is below 0.41), then one per span in time
    order: a missing span is exp(U) frames, U uniform between log(1 * fps) and log(60 * fps); a visible span is
    exponential with the mean mean_gap * (1 - 0.41) / 0.41, mean_gap = (60 - 1) * fps / (log(60 * fps) - log(1 * fps));
    each rounded and at least 1 frame. The spans alternate and the last one is cut at n."""
    low, high = math.log(1.0 * fps), math.log(60.0 * fps)
    mean_gap = (60.0 - 1.0) * fps / (high - low)
    mean_visible = mean_gap * (1.0 - 0.41) / 0.41
    flags = []
    missing = bool(rng.random() < 0.41)
    while len(flags) < n:
        if missing:
            length = max(1, int(round(math.exp(rng.uniform(low, high)))))
        else:
            length = max(1, int(round(rng.exponential(mean_visible))))
        flags += [missing] * length
        missing = not missing
    return flags[:n]


def spans_of(match):
    """{half: (kick-off row, stop row)}: the rows that a half's mask covers, from its kick-off to the block's end."""
    return {half: (match.kickoffs[half], stop) for half, _, stop in match.graded}


def late_kickoffs(match, first=40, second=30):
    """`match` with each half kicking off some rows after its graded block starts, as our second half does."""
    return dataclasses.replace(match, kickoffs={1: match.kickoffs[1] + first, 2: match.kickoffs[2] + second})


def owners_of(tracklets, n_frames):
    """(n_frames, 46) int: the number of the tracklet that holds each slot on each row, -1 where nobody does."""
    owner = np.full((n_frames, degrade.N_SLOTS), -1, dtype=np.int64)
    number = 0
    for half in sorted(tracklets):
        for t in tracklets[half]:
            assert (owner[t.start:t.stop, t.slot] == -1).all(), "two tracklets hold one slot on one row"
            owner[t.start:t.stop, t.slot] = number
            number += 1
    return owner


@contextlib.contextmanager
def no_calibration_load():
    """calibration.load raises while the block runs: whatever is done inside it loaded no Calibration of our match.

    Every module of the benchmark reaches the Calibration through this one function of this one module object (they
    import the module, not the function), so replacing it here replaces it for all of them."""
    real = calibration.load

    def refuse(*arguments, **keywords):
        raise AssertionError("calibration.load was called for the generic profile, which reads nothing of our match")
    calibration.load = refuse
    try:
        yield
    finally:
        calibration.load = real


def flipped_truth(match):
    """`match` with every flag of its truth turned over (alive and dead, home and away): the same tracking."""
    status, possession = np.array(match.ball_status), np.array(match.ball_possession)
    status[status >= 0] = 1 - status[status >= 0]
    possession[possession >= 0] = 1 - possession[possession >= 0]
    for array in (status, possession):
        array.setflags(write=False)
    return dataclasses.replace(match, ball_status=status, ball_possession=possession)


# -- the declared constants -------------------------------------------------------------------------------------------

def test_generic_constants_are_the_declared_ones():
    """The numbers of the profile are the declared ones, DATA.md still says the two figures taken from it, and the
    arithmetic that the design quotes follows from them.

    A constant that moved is not a tuning knob that somebody turned: it is a change of what the benchmark is, which the
    design forbids after a result was seen. So each is pinned, and the message says where it is declared and why it may
    not move. The two figures that are the organisers' are looked up in their document, in the words the comments at
    the constants quote, so that a new version of DATA.md is read again before anything follows it."""
    assert degrade.GENERIC == "generic" and degrade.PROFILES["generic"] == 1, (
        "the generic profile is version 1; a change of what it does is a new version. " + DECLARED)
    assert degrade.COMPONENTS["generic_gaps"] == 5, (
        "the stream of the generic profile's gaps is number 5 in the key of rng_for: a stream's number never changes")
    assert degrade.GENERIC_MISSING_SHARE == 0.41, "the share of the frames without a ball: " + DECLARED
    assert degrade.GENERIC_GAP_SECONDS == (1.0, 60.0), "the range of a gap of the ball, in seconds: " + DECLARED
    assert degrade.GENERIC_PRESENT_MEAN == 20.5, "the mean number of tracked objects: " + DECLARED
    assert degrade.GENERIC_TRACKED_SECONDS == (2.0, 300.0), "the range of a tracked span, in seconds: " + DECLARED
    assert degrade.GENERIC_TRACKED_GRID == 1024, "the number of tracked lengths that a draw is one of: " + DECLARED
    assert degrade.GENERIC_SMOOTH_WINDOW == 7, "the moving average's window, in frames: " + DECLARED
    assert (degrade.HOME_SLOTS, degrade.AWAY_SLOTS, degrade.N_SLOTS, degrade.FULL) == (24, 22, 46, 22), (
        "the slots are the file format's: 46, in pools of 24 and 22, for 22 players on the pitch")

    # The two published figures, and the 46 slots, in the organisers' own words.
    published = " ".join(read(os.path.join(ROOT, "DATA.md")).split())
    for sentence in ("41% of the graded ones", "the mean over graded frames is 20.5", "46 track slots"):
        assert sentence in published, (
            "DATA.md no longer says {!r}, which a constant of the generic profile was taken from: read what the "
            "organisers changed before anything else. {}".format(sentence, DECLARED))

    # What the design works out from them, by arithmetic: 6.8% of an on-pitch player's frames lost; a gap with a median
    # of 7.7 s, a mean of 14.4 s and a 90th percentile of 40 s; a lost span of 4.4 s on average.
    a, b = degrade.GENERIC_GAP_SECONDS
    assert round(100 * (1 - degrade.GENERIC_PRESENT_MEAN / 22), 1) == 6.8
    assert round(math.sqrt(a * b), 1) == 7.7 and round((b - a) / math.log(b / a), 1) == 14.4
    assert round(a * (b / a) ** 0.9) == 40
    low, high = degrade.GENERIC_TRACKED_SECONDS
    lost = 1 - degrade.GENERIC_PRESENT_MEAN / 22
    assert round((high - low) / math.log(high / low) * lost / (1 - lost), 1) == 4.4


# -- no Calibration ---------------------------------------------------------------------------------------------------

def test_generic_takes_no_calibration_and_refuses_one():
    """generic and clean read nothing of our match: needs_calibration says so, the degrader refuses a Calibration that
    is handed to generic and loads none, and the profile's own functions name nothing of the Calibration.

    NEEDS_CALIBRATION is the profiles that are fitted on the provided game (gaps, tracking, context), which nothing is
    trained or chosen on since the organisers' reply of 5 Nov 2026; with clean and generic they are all the profiles
    there are, so that a new profile has to be put on one side or the other."""
    assert degrade.NEEDS_CALIBRATION == frozenset({"gaps", "tracking", "context"})
    assert set(degrade.NEEDS_CALIBRATION) | {"clean", "generic"} == set(degrade.PROFILES), (
        "every profile either reads the Calibration of our match or is clean or generic: say which a new one is")
    assert not set(degrade.NEEDS_CALIBRATION) & {"clean", "generic"}
    for profile in ("clean", "generic"):
        assert degrade.needs_calibration(profile) is False, profile
    for profile in ("gaps", "tracking", "context"):
        assert degrade.needs_calibration(profile) is True, profile
    assert benchmark_run.needs_calibration is degrade.needs_calibration, "the runner asks the degrader's own rule"
    assert not degrade.needs_calibration(benchmark_run.DEFAULT_PROFILE), "the default reads nothing of our match"

    view = degrade.tracking_view(steady_match(rows=LONG_ROWS))
    with no_calibration_load():
        atd, info = degrade.degrade(view, "generic", None, 0)
    assert info["profile"] == "generic" and info["version"] == 1 and atd.n_slots == 46
    for given in (made_up_calibration(), tracking_calibration(), object()):
        expect_error(ValueError, lambda given=given: degrade.degrade(view, "generic", given, 0), "generic",
                     "reads nothing of our match", "takes no Calibration", "one was given")
    # The profiles that are fitted on our match still say that they need it.
    for profile in ("gaps", "tracking", "context"):
        expect_error(ValueError, lambda profile=profile: degrade.degrade(view, profile, None, 0), profile,
                     "Calibration", "calibration.load()")

    # No Calibration can be handed to the profile's own functions, and their code names none: no name, attribute or
    # argument of them is a calibration (their docstrings say that none is read, and are left out).
    assert list(inspect.signature(degrade.generic_parameters).parameters) == ["fps"]
    assert list(inspect.signature(degrade.generic_mask).parameters) == ["rng", "n", "fps"]
    assert list(inspect.signature(degrade._generic_ball).parameters) == ["view", "atd", "seed"]
    with open(degrade.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("generic_parameters", "generic_mask", "_generic_ball"):
        named = []
        for node in ast.walk(functions[name]):
            text = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                    else node.arg if isinstance(node, (ast.arg, ast.keyword)) else None)
            if text and "calibration" in text.lower():
                named.append(text)
        assert not named, "{} reaches for the Calibration of our match: {}".format(name, named)


def test_generic_no_instrument_loads_a_calibration():
    """The trainers and the instruments run the generic profile with no Calibration: none is loaded, and one that is
    given is refused.

    calibration.load raises for as long as each of them runs (no_calibration_load), on the synthetic matches: the two
    trainers' rows (train_dead_ball.gather and train_shape.gather, whose provenance then has no digest) and the rules'
    counts, the conventions instrument and the pipeline's latency, diagnose (whose report holds None for the
    Calibration), touch_stats and the restarts' anatomy. A Calibration handed to any of them with this profile is
    refused: by the trainers and diagnose themselves (BenchmarkError), and for the rest by the degrader that they
    serve the match through (a ValueError that names the profile)."""
    matches, cal = synthetic_matches(), synthetic_calibration()
    refused = benchmark_run.BenchmarkError
    with tempfile.TemporaryDirectory() as root:
        hmm = write_text(root, "hmm_only.toml", HMM_CONFIG)
        with no_calibration_load():
            for trainer in (train_dead_ball, train_shape):
                gathered = trainer.gather("generic", 2, 7, matches=matches, data_dir=root)
                assert gathered.calibration_sha256 is None and gathered.seeds == 2, trainer.__name__
                assert sorted(gathered.data) == ["RND001", "RND002"], trainer.__name__
            counts = train_dead_ball.measure_rules("generic", 1, matches=matches, data_dir=root)
            assert counts, "the rules' triggers were counted on the generic profile"
            inputs = train_dead_ball.served_inputs(matches["RND001"], "generic", 1, None, root)
            assert inputs.profile == "generic" and inputs.seed == 1 and inputs.atd.n_slots == 46
            assert leaks(inputs, matches["RND001"]) == []
            lags = conventions.measure(profile="generic", seeds=2, matches=matches, data_dir=root, log=quiet)
            assert lags["profile"] == "generic" and lags["seeds"] == [0, 1]
            latency = conventions.measure_latency(load_config(yardstick_path(0)), profile="generic", seeds=1,
                                                  matches=matches, data_dir=root, log=quiet)
            assert latency["profile"] == "generic" and latency["seeds"] == [0]
            report = diagnose.diagnose(hmm, profile="generic", seeds=1, matches=matches, data_dir=root, log=quiet)
            assert report["profile"] == {"name": "generic", "version": 1} and report["seeds"] == [0]
            assert report["tracking_calibration"] is None, "no Calibration made the tracking, and the report says so"
            touched = touch_stats.measure([TOUCH_SETTING], profile="generic", seeds=1, matches=matches, data_dir=root,
                                          log=quiet)
            assert touched["settings"], "touch_stats measured its setting on the generic profile"
            restarts.anatomy(yardstick_path(50), profile="generic", seeds=1, matches=matches, data_dir=root, log=quiet)

        # One that is given is refused, and never silently unused.
        for trainer in (train_dead_ball, train_shape):
            expect_error(refused, lambda trainer=trainer: trainer.gather("generic", 1, 7, matches=matches,
                                                                         calibration=cal, data_dir=root),
                         "generic", "takes no calibration", "one was given")
        expect_error(refused, lambda: diagnose.diagnose(hmm, profile="generic", seeds=1, matches=matches,
                                                        calibration=cal, data_dir=root, log=quiet),
                     "generic", "takes no calibration", "ignored")
        for call in (lambda: conventions.measure(profile="generic", seeds=1, matches=matches, calibration=cal,
                                                 data_dir=root, log=quiet),
                     lambda: train_dead_ball.measure_rules("generic", 1, matches=matches, calibration=cal,
                                                           data_dir=root),
                     lambda: touch_stats.measure([TOUCH_SETTING], profile="generic", seeds=1, matches=matches,
                                                 calibration=cal, data_dir=root, log=quiet),
                     lambda: restarts.anatomy(yardstick_path(50), profile="generic", seeds=1, matches=matches,
                                              calibration=cal, data_dir=root, log=quiet)):
            expect_error(ValueError, call, "generic", "one was given")


def test_generic_is_held_to_no_fidelity_gate_against_our_match():
    """The fidelity report measures a profile against our match's Calibration; generic is not built to pass it and is
    never run through it (the design of P27: "No fidelity gate against our match").

    The gates are those of the profiles that read the Calibration, and of no other. A fidelity report of generic is
    refused, by name and with the profiles that have gates, before a Calibration is loaded (calibration.load raises
    here) and before a match is (the loader raises too); the command says FAILED, exits 1 and writes nothing.

    What is held here is the refusal, not its reason. Until R030 fidelity.py's sentence gave the clean profile's reason
    for every profile without gates ("degrades nothing"), which is not true of generic (it degrades, and is not held
    to our match), and its --profile defaulted to the runner's default, so a bare `fidelity.py` asked for this
    refusal. R030 fixed both: the sentence now gives each profile's reason, and fidelity.DEFAULT_PROFILE is tracking.
    Only the default is asserted here."""
    assert set(fidelity.GATES) == set(degrade.NEEDS_CALIBRATION) == {"gaps", "tracking", "context"}, (
        "a gate holds a profile to a measurement of our match: only the profiles that read it have any")
    assert "generic" not in fidelity.GATES and "clean" not in fidelity.GATES
    assert fidelity.DEFAULT_PROFILE in fidelity.GATES and fidelity.DEFAULT_PROFILE != benchmark_run.DEFAULT_PROFILE, (
        "a bare fidelity.py measures a profile that has gates, not the runner's default (generic has none)")

    def no_match(match_id):
        raise AssertionError("a match was loaded for a fidelity report that is refused: " + match_id)

    with tempfile.TemporaryDirectory() as root, no_calibration_load(), loading(no_match):
        expect_error(fidelity.FidelityError,
                     lambda: fidelity.report("generic", seeds=1, matches=synthetic_matches(), data_dir=root, log=quiet),
                     "generic", "no gates", "context", "gaps", "tracking")
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = fidelity.main(["--profile", "generic", "--seeds", "1", "--data-dir", root])
        assert code == 1 and shown.getvalue().startswith("FAILED") and "generic" in shown.getvalue(), shown.getvalue()
        assert not os.path.exists(os.path.join(root, "runs")), "a refused report writes nothing"


# -- no truth ---------------------------------------------------------------------------------------------------------

def test_generic_never_sees_the_truth():
    """The degrader is handed a view with no flag of the truth, and what it makes does not depend on the truth: two
    matches that differ only in their flags get the same bytes and the same record, and the inputs that a degraded
    match is served with hold nothing of the match or its flags, to any depth."""
    names = [field.name for field in dataclasses.fields(degrade.TrackingView)]
    assert not [name for name in names if "status" in name.lower() or "possession" in name.lower()], names
    match = steady_match(rows=LONG_ROWS)
    view = degrade.tracking_view(match)
    assert not hasattr(view, "ball_status") and not hasattr(view, "ball_possession")
    assert not [held for held in vars(view) if "status" in held or "possession" in held]
    first, info = degrade.degrade(view, "generic", None, 0)

    other = flipped_truth(match)
    assert not np.array_equal(truth.truth_codes(other)[0], truth.truth_codes(match)[0]), "the two truths differ"
    for name in ("players_xy", "players_speed", "player_team", "ball_xy", "ball_z", "ball_speed"):
        assert getattr(other, name) is getattr(match, name), "and nothing else does: " + name
    again, info_again = degrade.degrade(degrade.tracking_view(other), "generic", None, 0)
    assert everything(again) == everything(first), "the truth changed, and so did the degraded tracking"
    assert info_again == info

    inputs = match_inputs.MatchInputs(match, first, profile="generic", seed=0)
    assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"
    assert leaks(view, match) == [], "the view itself holds none either"
    assert inputs.profile == "generic" and inputs.atd is first and inputs.atd.n_slots == 46
    assert inputs.atd_manifest["profile"] == "generic"
    assert "array" not in repr(first), "no frame in a repr"


# -- the draws --------------------------------------------------------------------------------------------------------

def test_generic_is_deterministic_and_independent_of_the_other_matches():
    """The same seed gives the same arrays and the same record; another seed, and another match_index, give another
    ball and other players; and a match gets the same degradation whichever matches are run with it, in whatever
    order, and alone."""
    matches = {"RND001": steady_match("RND001", 0, rows=LONG_ROWS),
               "RND002": steady_match("RND002", 1, rows=(2800, 3100)),
               "J03WMX": steady_match("J03WMX", 2, rows=LONG_ROWS)}

    def keyed(order):
        return {(seed, match_id): everything(degrade.degrade(degrade.tracking_view(matches[match_id]), "generic",
                                                             None, seed)[0])
                for seed in (0, 1) for match_id in order}

    forward, backward = keyed(["RND001", "RND002", "J03WMX"]), keyed(["J03WMX", "RND002", "RND001"])
    assert forward == backward, "the order of the matches changed a degradation"
    assert keyed(["RND002"])[(1, "RND002")] == forward[(1, "RND002")], "alone, a match gets what it got in the run"
    assert len(set(forward.values())) == len(forward), "the matches and the seeds draw apart"

    match = matches["RND001"]
    view = degrade.tracking_view(match)
    first, info = degrade.degrade(view, "generic", None, 0)
    again, info_again = degrade.degrade(degrade.tracking_view(match), "generic", None, 0)
    assert everything(first) == everything(again) and info == info_again
    for name in ARRAYS:
        assert not getattr(first, name).flags.writeable, name + " is read-only"
    other_seed, _ = degrade.degrade(view, "generic", None, 1)
    elsewhere, _ = degrade.degrade(degrade.tracking_view(match, match_index=3), "generic", None, 0)
    for label, other in (("seed 1 is not seed 0", other_seed), ("another match_index is another stream", elsewhere)):
        assert other.ball_xy.tobytes() != first.ball_xy.tobytes(), "the ball: " + label
        assert other.players_xy.tobytes() != first.players_xy.tobytes(), "the players: " + label
    assert degrade.tracking_view(matches["J03WMX"]).match_index == 0 and view.match_index == degrade.index_of("RND001")
    # Over many seeds the masks and the tracklets are many: the draws are not stuck on a few.
    drawn = [degrade.degrade(view, "generic", None, seed)[0] for seed in range(12)]
    assert len({atd.ball_xy.tobytes() for atd in drawn}) == 12 and len({players_bytes(atd) for atd in drawn}) == 12

    # The record of what was drawn is JSON-ready, and holds counts and shares: no frame.
    assert sorted(info) == INFO_KEYS, sorted(info)
    assert (info["profile"], info["version"], info["seed"], info["match_id"]) == ("generic", 1, 0, "RND001")
    assert info["match_index"] == view.match_index
    assert json.loads(json.dumps(info, sort_keys=True)) == json.loads(json.dumps(info_again, sort_keys=True))
    assert sorted(info["ball"]) == ["gap_seconds", "halves", "missing_share_target"]
    assert info["ball"]["missing_share_target"] == 0.41 and info["ball"]["gap_seconds"] == [1.0, 60.0]
    assert sorted(info["ball"]["halves"]) == sorted(info["players"]["halves"]) == [1, 2]
    assert all(sorted(record) == ["gaps", "missing_frames", "rows"] for record in info["ball"]["halves"].values())
    assert "transplants" not in info and "tracking_version" not in info, "nothing of ours was transplanted"


def test_generic_mask_is_the_documented_draws():
    """generic_mask is the documented sequence of draws, to the frame; its gaps are log-uniform between 1 s and 60 s,
    its visible spans exponential, and over many long draws the ball is missing on 41% of the frames.

    Against reference_mask, which restates the documentation one frame at a time with the declared numbers written
    out, from a generator of the same key: lengths from none to 20,000 frames at three rates (at 1 Hz a gap is 1 to 60
    frames, so the floor of one frame and the rounding show). A longer mask of the same generator starts with the
    shorter one: the draws are made in time order and the last span is cut.

    Then 60 draws of 70,000 frames at 25 Hz (a half is about that long), 4,789 whole gaps: every one is between 25
    and 1,500 frames and both ends are reached; their median, mean and 90th percentile are the law's (sqrt(25 * 1500) =
    193.6, 59 * 25 / ln 60 = 360.3 and 25 * 60 ** 0.9 = 996.0 frames) to within 10%, 5% and 8%, each more than three
    standard errors (5.7, 5.5 and 17.7 frames); the 4,750 whole visible spans between two gaps have the mean 360.25 *
    0.59 / 0.41 = 518.4 frames to within 5% (standard error 7.5) and 63.2% of them are shorter than it, as an
    exponential's are; and the pooled missing share is 0.41 to within 0.02. One half alone is not: its share has a
    standard deviation of about 0.04 (these 60 run from 0.31 to 0.49), which is the law's and no error. The first
    frame is missing in 41% of 2,000 masks, to within 0.04 (standard error 0.011)."""
    for fps in (25, 10, 1):
        for n in (0, 1, 2, 60, 1000, 20000):
            key = [5, n, fps]
            ours = degrade.generic_mask(np.random.default_rng(key), n, fps)
            assert ours.dtype == np.bool_ and ours.shape == (n,), (fps, n)
            assert ours.tolist() == reference_mask(np.random.default_rng(key), n, fps), (fps, n)
        long = degrade.generic_mask(np.random.default_rng([6, fps]), 20000, fps)
        short = degrade.generic_mask(np.random.default_rng([6, fps]), 700, fps)
        assert short.tolist() == long[:700].tolist(), "a shorter mask is the start of a longer one"
        assert long.any() and not long.all(), "a mask of 20,000 frames has gaps, and frames with the ball"
        whole = [stop - first for first, stop in runs_of(long.tolist()) if stop < long.size]
        assert whole and min(whole) >= round(1.0 * fps) and max(whole) <= round(60.0 * fps), (fps, min(whole))
    empty = degrade.generic_mask(np.random.default_rng(0), 0, 25)
    assert empty.shape == (0,) and empty.dtype == np.bool_ and empty.tolist() == []

    fps, frames = 25, 70000
    shortest, longest = round(1.0 * fps), round(60.0 * fps)
    assert (shortest, longest) == (25, 1500)
    gaps, visible, missing, shares = [], [], 0, []
    for draw in range(60):
        mask = degrade.generic_mask(np.random.default_rng([11, draw]), frames, fps)
        flags = mask.tolist()
        gaps += [stop - first for first, stop in runs_of(flags) if stop < frames]         # the last may be cut at n
        visible += [stop - first for first, stop in runs_of([not flag for flag in flags])
                    if first > 0 and stop < frames]                                       # between two gaps
        missing += int(mask.sum())
        shares.append(float(mask.mean()))
    gaps, visible = np.array(gaps), np.array(visible)
    assert gaps.size > 4000 and visible.size > 4000, (gaps.size, visible.size)
    assert gaps.min() >= shortest and gaps.max() <= longest, (gaps.min(), gaps.max())
    assert gaps.min() < 1.2 * shortest and gaps.max() > longest / 1.2, "both ends of the declared range are reached"
    assert visible.min() >= 1, "two gaps never touch: there is a frame with the ball between them"
    law_median, law_mean, law_p90 = math.sqrt(shortest * longest), 59 * fps / math.log(60), shortest * 60 ** 0.9
    assert abs(np.median(gaps) / law_median - 1) < 0.10, (np.median(gaps), law_median)
    assert abs(gaps.mean() / law_mean - 1) < 0.05, (gaps.mean(), law_mean)
    assert abs(np.quantile(gaps, 0.9) / law_p90 - 1) < 0.08, (np.quantile(gaps, 0.9), law_p90)
    law_visible = law_mean * 0.59 / 0.41
    assert abs(visible.mean() / law_visible - 1) < 0.05, (visible.mean(), law_visible)
    assert abs((visible < law_visible).mean() - (1 - math.exp(-1))) < 0.03, (visible < law_visible).mean()
    pooled = missing / (60 * frames)
    assert abs(pooled - 0.41) < 0.02, pooled
    assert min(shares) < 0.36 and max(shares) > 0.46, "one half's share is far from 0.41 as often as the law says"

    first_missing = sum(bool(degrade.generic_mask(np.random.default_rng([12, k]), 1, fps)[0]) for k in range(2000))
    assert abs(first_missing / 2000 - 0.41) < 0.04, first_missing

    rng = np.random.default_rng(0)
    for bad in (-1, 2.5, True, None):
        expect_error(ValueError, lambda bad=bad: degrade.generic_mask(rng, bad, 25), "n", "at least 0")
    for bad in (0, -25, 2.5, True):
        expect_error(ValueError, lambda bad=bad: degrade.generic_mask(rng, 100, bad), "fps")


# -- the ball ---------------------------------------------------------------------------------------------------------

def test_generic_ball_is_missing_exactly_where_the_masks_say():
    """The ball is gone, x and y together, exactly where each half's mask is True, from the half's kick-off to the end
    of its graded block, and is DFL's own everywhere else; its speed is NaN on every row; the record's counts are the
    masks'.

    The masks are reference_mask's from np.random.default_rng([seed, match_index, 5, half]), the "generic_gaps" stream:
    the profile's own, so the ball is the same whatever the players are (another squad, other positions), and wherever
    the ball itself is (the gaps are blind to the play), and the players are the same whatever the ball is. Each half
    here kicks off some rows after its block starts, so that a mask laid from the block's start would be seen."""
    match = late_kickoffs(steady_match(rows=(6000, 5000)))
    view = degrade.tracking_view(match)
    spans = spans_of(match)
    assert all(kickoff > start for (_, start, _), (kickoff, _) in zip(match.graded, spans.values()))
    before = {name: getattr(match, name).tobytes() for name in ("players_xy", "ball_xy", "ball_speed", "ball_z")}
    inside = np.zeros(match.n_frames, dtype=bool)
    for kickoff, stop in spans.values():
        inside[kickoff:stop] = True
    total_gaps = 0
    for seed in (0, 1, 2):
        atd, info = degrade.degrade(view, "generic", None, seed)
        assert atd.ball_xy.shape == (match.n_frames, 2) and atd.ball_speed.shape == (match.n_frames,)
        assert np.isnan(atd.ball_speed).all(), "the profile promises no ball speed, on any row"
        gone = np.isnan(atd.ball_xy)
        assert (gone[:, 0] == gone[:, 1]).all(), "x and y are missing together"
        covered = missing = 0
        for half, (kickoff, stop) in spans.items():
            flags = reference_mask(np.random.default_rng([seed, view.match_index, 5, half]), stop - kickoff, match.fps)
            assert gone[kickoff:stop, 0].tolist() == flags, (seed, half, "missing exactly where the mask says")
            assert info["ball"]["halves"][half] == {"rows": stop - kickoff, "gaps": len(runs_of(flags)),
                                                    "missing_frames": sum(flags)}, (seed, half)
            covered, missing = covered + (stop - kickoff), missing + sum(flags)
            total_gaps += len(runs_of(flags))
        assert abs(info["mask_share"] - missing / covered) < 1e-12, "the share of the masks, over the rows they cover"
        seen = ~gone[:, 0]
        assert np.array_equal(atd.ball_xy[seen], match.ball_xy[seen]), "what is seen is DFL's"
        assert np.array_equal(atd.ball_xy[~inside], match.ball_xy[~inside], equal_nan=True), (
            "the rows before a kick-off and between the halves are not touched")
        assert not atd.ball_xy.flags.writeable and not atd.ball_speed.flags.writeable
        assert atd.ball_present.tolist() == seen.tolist()
    assert total_gaps > 20, "the masks of these halves have gaps to place: {}".format(total_gaps)
    assert {name: getattr(match, name).tobytes() for name in before} == before, "the match was not touched"

    # The ball's draws are its own stream's: another squad on other positions, and another ball, under the same id.
    first, info = degrade.degrade(view, "generic", None, 0)
    other = late_kickoffs(steady_match(seed=7, rows=(6000, 5000), teams=(0,) * 4 + (1,) * 5, substitutions=False))
    assert other.match_id == match.match_id and other.players_xy.shape != match.players_xy.shape
    assert not np.array_equal(other.ball_xy, match.ball_xy, equal_nan=True)
    elsewhere, info_elsewhere = degrade.degrade(degrade.tracking_view(other), "generic", None, 0)
    assert np.array_equal(np.isnan(elsewhere.ball_xy), np.isnan(first.ball_xy)), "the same mask, whoever plays"
    assert info_elsewhere["ball"] == info["ball"] and info_elsewhere["mask_share"] == info["mask_share"]
    assert info_elsewhere["players"] != info["players"]
    # And the players' draws do not look at the ball.
    moved = dataclasses.replace(match, ball_xy=other.ball_xy)
    assert players_bytes(degrade.degrade(degrade.tracking_view(moved), "generic", None, 0)[0]) == players_bytes(first)

    # A ball that DFL itself has no row for stays missing, and is not in the mask's share: ten rows that the mask
    # leaves visible are taken out of DFL's ball, and the ball is missing on exactly ten rows more than the mask says.
    kickoff, stop = spans[1]
    holes = kickoff + np.flatnonzero(~np.isnan(first.ball_xy[kickoff:stop, 0]))[:10]
    assert holes.size == 10
    holed = np.array(match.ball_xy)
    holed[holes] = np.nan
    holed.setflags(write=False)
    atd, info_holed = degrade.degrade(degrade.tracking_view(dataclasses.replace(match, ball_xy=holed)), "generic",
                                      None, 0)
    assert np.isnan(atd.ball_xy[holes]).all(), "a DFL ball that was missing stays missing, whatever the mask says"
    assert info_holed["mask_share"] == info["mask_share"] and info_holed["ball"] == info["ball"]
    assert int(np.isnan(atd.ball_xy[:, 0]).sum()) == int(np.isnan(first.ball_xy[:, 0]).sum()) + 10


# -- the players ------------------------------------------------------------------------------------------------------

def test_generic_parameters_are_the_declared_rules():
    """generic_parameters is the declared rules and nothing else: d = 1 - 20.5 / 22, 1,024 tracked lengths at the
    mid-quantiles of a log-uniform law between 2 s and 300 s, a 7-frame moving average, and none of the parts that
    DATA.md does not document (no wrong-team episode, no NaN-speed run, no shortest piece, no rescale).

    The lengths are worked out here from the declared numbers, written out: length i is exp(ln 2 + (ln 300 - ln 2) * (i
    + 0.5) / 1024) seconds, in frames at the match's rate, rounded and at least 1. The mean lost span follows: mu = mean
    tracked * d / (1 - d), 4.35 s at 25 Hz (the design's 4.4 s)."""
    parameters = degrade.generic_parameters(25)
    assert isinstance(parameters, degrade.TrackingParameters)
    assert parameters.d == 1 - 20.5 / 22 and abs(parameters.d - 0.0681818) < 1e-6
    assert parameters.smooth == 7 and parameters.min_piece == 0 and parameters.tracked_scale == 1.0
    assert parameters.episodes.size == 0 and parameters.nan_runs.size == 0
    assert (parameters.scale, parameters.wrong_team_share, parameters.speed_nan_share, parameters.mean_episode) == (
        0.0, 0.0, 0.0, 0.0) and parameters.onset_rate == 0.0
    lengths = parameters.lengths
    assert lengths.shape == (1024,) and lengths.dtype == np.int64 and not lengths.flags.writeable
    expected = [max(1, round(math.exp(math.log(2.0) + (math.log(300.0) - math.log(2.0)) * (i + 0.5) / 1024) * 25))
                for i in range(1024)]
    assert lengths.tolist() == expected, "the lengths are the documented formula's"
    assert lengths.min() >= round(2 * 25) and lengths.max() <= round(300 * 25), (lengths.min(), lengths.max())
    assert (lengths.min(), lengths.max()) == (50, 7482), "the mid-quantiles: the first is 2 s, the last just under 300"
    assert (np.diff(lengths) >= 0).all(), "in the order of the quantiles"
    assert abs(np.median(lengths) / 25 - math.sqrt(2 * 300)) < 0.05, "log-uniform: the median is the geometric mean"
    law_mean = (300 - 2) * 25 / math.log(300 / 2)
    assert parameters.mean_tracked == sum(expected) / 1024 and abs(parameters.mean_tracked / law_mean - 1) < 1e-3
    assert parameters.mu == parameters.mean_tracked * parameters.d / (1 - parameters.d)
    assert abs(parameters.mu / 25 - 4.35) < 0.01, parameters.mu / 25
    # Another rate: the same seconds in its own frames. And no rate that is not a whole number of at least 1.
    doubled = degrade.generic_parameters(50)
    assert doubled.d == parameters.d and doubled.smooth == 7
    assert (doubled.lengths.min(), doubled.lengths.max()) == (100, 14963)
    assert np.abs(doubled.lengths - 2 * lengths).max() <= 1, "rounded after the rate, not before"
    for bad in (0, -25, 2.5, True):
        expect_error(ValueError, lambda bad=bad: degrade.generic_parameters(bad), "fps")


def test_generic_players_are_lost_at_the_declared_rate_in_46_slots():
    """On a long full-strength match the players are lost at the declared rate, in the tracker's 46 slots, and are what
    the players' machinery makes of generic_parameters with no cap: nothing else is in them.

    22 players on the pitch all through two halves of 30,000 rows, four seeds. Each is lost on d = 1 - 20.5 / 22 of his
    frames in expectation, so 20.5 slots are present on average: the mean over the graded frames of one seed is within
    0.4 of it and the mean of the four within 0.2 (one seed's mean has a standard deviation of about 0.08: each
    player has some 19 tracked and lost spans a half, of very unequal lengths). The slots are 24 home then 22 away and
    a slot carries only players of its pool's team; no slot changes player between two adjacent rows, and slots are
    taken again (after at least one empty row); a tracklet's speed is NaN on its first row and finite on every other;
    the positions are the player's own under a centred 7-frame moving average inside the piece, and the speed the step
    of those in m/s. No wrong-team episode, no NaN-speed run, no capped frame and no dropped piece is counted."""
    match = steady_match(rows=SQUAD_ROWS, teams=SQUAD, substitutions=False)
    view = degrade.tracking_view(match)
    parameters = degrade.generic_parameters(match.fps)
    graded = np.zeros(match.n_frames, dtype=bool)
    for _, start, stop in match.graded:
        graded[start:stop] = True
    on_pitch = int((~np.isnan(match.players_xy[:, :, 0])).sum())
    assert on_pitch == 22 * int(graded.sum()), "everybody is on the pitch on every graded row"
    length, width = match.pitch
    means, shared, taken_again = [], 0, 0
    for seed in range(4):
        atd, info = degrade.degrade(view, "generic", None, seed)
        assert atd.slot_team.dtype == np.int8 and atd.slot_team.tolist() == [0] * 24 + [1] * 22
        assert atd.players_xy.shape == (match.n_frames, 46, 2) and atd.players_speed.shape == (match.n_frames, 46)
        assert atd.n_slots == 46 and not atd.players_xy.flags.writeable and not atd.players_speed.flags.writeable
        present = atd.players_present
        assert not present[~graded].any(), "between the halves there are no players, as in DFL"
        means.append(float(present[graded].sum(axis=1).mean()))
        assert abs(means[-1] - 20.5) < 0.4, (seed, means[-1])

        # The players are exactly the machinery's on the declared parameters, with no cap: the same bytes.
        xy, speed, tracklets, _ = degrade.degraded_players(view, parameters, seed)
        assert xy.tobytes() == atd.players_xy.tobytes() and speed.tobytes() == atd.players_speed.tobytes()
        records = [t for half in sorted(tracklets) for t in tracklets[half]]
        assert len(records) == info["players"]["tracklets"] > 500
        assert int(present.sum()) == sum(t.stop - t.start for t in records), "every present slot-frame has one tracklet"
        for t in records:
            assert (t.slot < 24) == (int(match.player_team[t.player]) == 0), "a slot carries its pool's team: {!r}".format(t)
            assert np.isnan(atd.players_speed[t.start, t.slot]), "no speed on the first row of a tracklet"
        owner = owners_of(tracklets, match.n_frames)
        both = (owner[1:] >= 0) & (owner[:-1] >= 0)
        assert not (both & (owner[1:] != owner[:-1])).any(), "a slot changed player between two adjacent rows"
        for half in sorted(tracklets):
            last = {}
            for t in sorted(tracklets[half], key=lambda t: (t.slot, t.start)):
                if t.slot in last:
                    shared += 1
                    taken_again += t.start == last[t.slot] + 1
                    assert t.start > last[t.slot], "an empty row between two tracklets of a slot"
                last[t.slot] = t.stop
        # The only NaN speeds are the tracklets' first rows: no run of them is placed.
        assert int((present & np.isnan(atd.players_speed)).sum()) == len(records)
        assert np.isnan(atd.players_speed[~present]).all(), "no speed where nobody is"

        # Smoothed over 7 frames inside the piece, and the speed is the step of the smoothed positions.
        checked = 0
        for t in [t for t in records if 8 <= t.stop - t.start <= 400][:8]:
            run = smoothed_plain([tuple(row) for row in match.players_xy[t.start:t.stop, t.player]], 7)
            shown = atd.players_xy[t.start:t.stop, t.slot]
            assert np.allclose(shown, run, rtol=0, atol=1e-12), repr(t)
            assert not np.allclose(shown, match.players_xy[t.start:t.stop, t.player], rtol=0, atol=1e-6), (
                "the average moved nothing: " + repr(t))
            for row in range(1, t.stop - t.start):
                step = math.hypot((shown[row, 0] - shown[row - 1, 0]) * length,
                                  (shown[row, 1] - shown[row - 1, 1]) * width)
                assert math.isclose(atd.players_speed[t.start + row, t.slot], step * match.fps, rel_tol=1e-9), repr(t)
            checked += 1
        assert checked == 8, checked

        players = info["players"]
        assert players["d"] == round(1 - 20.5 / 22, 9) and players["smooth_window"] == 7
        assert players["tracked_scale"] == 1.0 and players["min_piece_frames"] == 0 and players["scale"] == 0.0
        assert players["wrong_team_target"] == 0.0 and players["speed_nan_target"] == 0.0
        for name in ("onsets", "episodes", "episode_frames", "speed_nan_runs", "capped_frames", "cap_slot_frames",
                     "cap_pieces_cut", "short_pieces", "short_slot_frames", "short_episode_pieces"):
            assert players[name] == 0, (name, players[name])
        assert players["speed_nan_frames"] == len(records), "the first row of each tracklet, and no other"
        assert players["on_pitch_frames"] == on_pitch and players["lost_frames"] == on_pitch - int(present.sum())
        assert abs(players["lost_share"] - (1 - present.sum() / on_pitch)) < 1e-9
        assert abs(players["lost_share"] - parameters.d) < 0.4 / 22
    assert abs(sum(means) / len(means) - 20.5) < 0.2, means
    assert shared > 1000 and taken_again > 0, (shared, taken_again)

    # The pools are the tracker's whatever the squads are, and a player off the pitch takes no slot.
    small = steady_match(rows=LONG_ROWS)
    atd, info = degrade.degrade(degrade.tracking_view(small), "generic", None, 0)
    assert atd.slot_team.tolist() == [0] * 24 + [1] * 22 and atd.players_xy.shape[1] == 46
    assert info["players"]["on_pitch_frames"] == int((~np.isnan(small.players_xy[:, :, 0])).sum())
    assert int(atd.players_present.sum()) == info["players"]["on_pitch_frames"] - info["players"]["lost_frames"]
    assert (atd.players_present.sum(axis=1) <= (~np.isnan(small.players_xy[:, :, 0])).sum(axis=1)).all()


# -- the runner -------------------------------------------------------------------------------------------------------

def independent_halves(match, config_path, seed, data_dir):
    """[HalfScore] of one config on one match under `generic`: degrade, predict and score_halves called here."""
    atd, _ = degrade.degrade(degrade.tracking_view(match), "generic", None, seed)
    inputs = match_inputs.MatchInputs(match, atd, profile="generic", seed=seed, data_dir=data_dir)
    codes, _ = benchmark_run.predict(load_config(config_path), inputs)
    return score.score_halves(truth.truth_codes(match)[0], codes, match.graded, match.match_id, seed)


def test_generic_runs_on_the_runner_with_no_calibration():
    """The runner serves each match's generic tracking, per seed, and the report says that no Calibration made it.

    Two seeds on the two synthetic matches, two configs: every half is the one that the degrader, predict and
    score_halves give when they are called here, a seed degrading the match with its own draws; the result and every
    report hold `calibration: None` and the profile's name and version; the printout names the profile and has no
    calibration line. The reports go to <data dir>/runs/benchmark/generic/, named as a degraded profile's are: `.s<n>`
    for a number of seeds that is not the default. And through the command line with no --profile: the default is this
    profile, and calibration.load raises for as long as it runs."""
    matches = synthetic_matches()
    paths = [yardstick_path(50), yardstick_path(0)]
    with tempfile.TemporaryDirectory() as root:
        lines = []
        with no_calibration_load():
            result = benchmark_run.evaluate_configs(paths, profile="generic", seeds=2, matches=matches, data_dir=root,
                                                    log=lines.append)
        assert result["profile"] == {"name": "generic", "version": 1} and result["seeds"] == [0, 1]
        assert result["calibration"] is None
        for entry, width in zip(result["configs"], (50, 0)):
            report = entry["report"]
            assert report["profile"] == {"name": "generic", "version": 1} and report["seeds"] == [0, 1]
            assert "calibration" in report and report["calibration"] is None, "no Calibration, and the report says so"
            assert "sensitivity_only" not in report, "the default is the yardstick, not a sensitivity check"
            expected = [half for match in matches.values() for seed in (0, 1)
                        for half in independent_halves(match, yardstick_path(width), seed, root)]
            assert len(report["halves"]) == 8 and [
                (h["match_id"], h["half"], h["seed"], h["n_gt"], h["n_pred"], tuple(h["tp"]),
                 tuple(tuple(row) for row in h["confusion"])) for h in report["halves"]] == [
                (h.match_id, h.half, h.seed, h.n_gt, h.n_pred, h.tp, h.confusion) for h in expected], width
        table = "\n".join(lines)
        assert "profile   generic (version 1)" in table and "seeds 0,1" in table, table
        assert not [line for line in lines if line.startswith("calibration") or line.startswith("note")], lines

        # The files: the profile's own folder, and a seed count in the name when it is not the default.
        default = benchmark_run.evaluate_configs(paths[:1], profile="generic", matches=matches, data_dir=root,
                                                 log=quiet)
        assert default["seeds"] == [0, 1, 2], "none asked for: DEFAULT_SEEDS, as every profile that degrades"
        one = benchmark_run.evaluate_configs(paths[:1], profile="generic", seeds=1, matches=matches, data_dir=root,
                                             log=quiet)
        written = [benchmark_run.write_reports(each)[0] for each in (result, default, one)]
        folder = os.path.join(root, "runs", "benchmark", "generic")
        subset = ".m-RND001-RND002"
        assert [os.path.relpath(path, folder) for path in written] == [
            "nearest_w50" + subset + ".s2.json", "nearest_w50" + subset + ".json", "nearest_w50" + subset + ".s1.json"]
        assert sorted(os.listdir(os.path.join(root, "runs", "benchmark"))) == ["generic"]
        stored = strict_json(read(written[0]))
        assert stored == result["configs"][0]["report"] and stored["calibration"] is None
        assert benchmark_run.report_path("d", "generic", "s", "raw", sources.MATCH_IDS).replace("\\", "/") == (
            "d/runs/benchmark/generic/s.json"), "all seven matches, the raw truth, the default seeds: the bare name"
        assert os.path.basename(benchmark_run.report_path("d", "generic", "s", "coarse", ["J03WMX"], 1)) == (
            "s.coarse.m-J03WMX.s1.json")

        # The profile does something: the same config on the clean tracking, and on another seed, scores differently.
        clean = benchmark_run.evaluate_configs(paths[:1], profile="clean", matches=matches, data_dir=root, log=quiet)
        assert clean["configs"][0]["report"]["pooled"] != one["configs"][0]["report"]["pooled"]
        assert clean["calibration"] is None and clean["seeds"] == [0]
        by_seed = {seed: [h for h in result["configs"][0]["report"]["halves"] if h["seed"] == seed] for seed in (0, 1)}
        assert [h["confusion"] for h in by_seed[0]] != [h["confusion"] for h in by_seed[1]], "two seeds, two draws"

    # The command line, with no --profile and no Calibration to be found: the default profile's folder.
    def serve(match_id):
        return random_match(match_id, seed=sources.MATCH_IDS.index(match_id))

    with tempfile.TemporaryDirectory() as root, loading(serve), no_calibration_load():
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = benchmark_run.main([yardstick_path(50), "--data-dir", root, "--matches", "J03WMX", "J03WN1",
                                       "--seeds", "1"])
        out = shown.getvalue()
        assert code == 0 and "profile   generic (version 1)" in out and "calibration" not in out, out
        assert os.listdir(os.path.join(root, "runs", "benchmark")) == ["generic"], out
        name = "nearest_w50.m-J03WMX-J03WN1.s1.json"
        assert os.listdir(os.path.join(root, "runs", "benchmark", "generic")) == [name] and name in out, out
        report = strict_json(read(os.path.join(root, "runs", "benchmark", "generic", name)))
        assert report["profile"] == {"name": "generic", "version": 1} and report["calibration"] is None
