"""Checks of selftrain (pipeline/benchmark/selftrain.py): P22(b)'s gate on the DFL matches and its label-free checks on
BAR-ATH (R027; the design's sections 6.1 and 6.4).

The gate decides whether the lane goes on, so what it measures is held against plain code: every number of its report
is worked out again here, frame by frame, from the same inputs and the estimators' own functions, on synthetic matches
whose touch rule gives enough rows for the fits (a stretch of play is a stretch of carrying; the 500-row floor is the
real one). What is held against it:

* the settings are `main.toml`'s (the teacher's rule, epsilon, the two models, the clip, Smart Tagging's c_team), and a
  config that lacks an entry is refused by name (test_selftrain_settings_read_main);
* R027's two configs (pipeline/benchmark/configs/r027/) are the base config (r024_both.toml: main.toml's entries today,
  and an adoption leaves it as it is) with its one `shape_possession` entry replaced by the declared `shape_selftrain`
  one and nothing else re-chosen, the second without the refiner
  (test_selftrain_r027_configs_are_the_base_with_the_source_swapped);
* rung 3's head is the penalised, class-balanced fit that the design declares, against scipy's L-BFGS-B on the same
  objective, then an intercept that is E2 on z plus the head; standardised as declared; the same bits whatever the
  order of the rows; refusals (test_selftrain_head_fit_matches_scipy, test_selftrain_head_is_standardised_...);
* the gate's variants, its two strata, the held-out touch agreement, the cross-fit offsets, the oracles and the
  teacher's figures equal a plain loop over the frames (test_selftrain_gate_equals_plain_loops); the base variant is
  `shape_possession`'s own `home`, and rung 1 with every offset forced to 0 is the base, to the bit
  (test_selftrain_gate_on_synthetic_matches);
* the oracles' arithmetic (E, B, the decomposition, the band) on hand-built cells, and the decision at its edges with
  the exit codes it gives (test_selftrain_oracle_arithmetic_by_hand, test_selftrain_gate_verdict_at_the_edges);
* no truth reaches the teacher or the offsets: a match with its truth mirrored (H for A) and one with its truth flipped
  on some stretches keep every teacher row, every offset and every fit as they were, and move what is measured against
  the truth (the losses, the accuracy, the teacher's precision); the oracles are fitted on the truth too, but the
  class-balanced intercept does not care which class is called home, so the mirror leaves them as they were and only
  the partial flip moves them; a match whose fold is missing is refused before it is loaded
  (test_selftrain_never_reads_truth_through_inputs, test_selftrain_gate_refuses_what_would_score_in_sample);
* the report is aggregates and numbers, with no list longer than its table (test_selftrain_report_is_aggregates_only);
* BAR-ATH's report: the offsets, the shares, the band and the Smart Tagging check worked out by a plain loop, the
  check's edge at the margin, and the exit codes, a failed check winning over the band
  (test_selftrain_bar_ath_band_and_tag_check, test_selftrain_tag_check_at_the_margin);
* the command line, its file names and its exit codes: an error is 2 and never a crash that looks like a verdict
  (test_selftrain_command_line_exit_codes).

Everything runs on arrays and synthetic matches made here: none reads data, and no number printed is a figure of any
match. The one fixture that takes a few seconds, the gate on two synthetic matches, is run once and shared; a check
that changes something copies it first.
"""

import atexit
import contextlib
import copy
import dataclasses
import functools
import io
import json
import math
import os
import shutil
import tempfile
import types

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import conventions, selftrain, train_shape, truth
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import CONFIG_FOLDER, loading, strict_json, synthetic_calibration
from pipeline.benchmark.dfl import DflMatch
from pipeline.benchmark.run import BenchmarkError
from pipeline.benchmark.train_dead_ball import served_inputs
from pipeline.config import load_config
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, recalibration, shape_features, shape_possession, smart_tagging
from pipeline.selftest import (SHAPE_COLUMN, expect_error, play_inputs, stump, touch_dead_model, write_model,
                               write_shape_model, write_text)

# Params
H, A = pm.H, pm.A
IDS = ("RND001", "RND002")
SCENE_ROWS, SCENE_GAP, SCENE_PITCH, SLOTS = (5600, 5200), 25, (105.0, 68.0), 12
MIN_PROB = 0.4                     # the clip binds on the frames where z is 0.8 (sigma 0.69), so the tallies see it
TEACHER = {"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1, "duel_radius": 1.0}
TOP_KEYS = {"kind", "profile", "seeds", "matches", "calibration", "settings", "constants", "git", "per_match",
            "pooled", "oracles", "gate"}
MATCH_KEYS = {"gate_frames", "variants", "hold", "touch", "oracle", "teacher", "fallbacks", "cells"}
VARIANTS = ("base", "rung1", "e1", "single", "rung3")
R027_CONFIGS = os.path.join(CONFIG_FOLDER, "r027")
# What R027's configs are made from, and what the check below holds them to (main.toml is the candidate once adopted)
BASE_CONFIG = os.path.join(selftrain.ROOT, "pipeline", "configs", "r024_both.toml")
# The entry that the design's section 2 declares for R027's candidate, as written there (the source's docstring has it
# too): the base's one `shape_possession` entry is replaced by it, comments included, and nothing else is touched.
R027_ENTRY = """[[evidence]]
kind = "shape_selftrain"
weight = 0.2
model = "pipeline/models/shape_possession_hgb.json"
scope = "all"
min_prob = 0.02
alive_model = "pipeline/models/dead_ball_hgb.json"     # the teacher's dead-ball mask
teacher = { radius = 1.5, max_rel_speed = 6.0, min_frames = 1, duel_radius = 1.0 }
epsilon = 0.07                                         # 1 - controlled_touch's confidence 0.93 (R013)"""


# -- a scene with enough touches for the fits -------------------------------------------------------------------------

def carry_scene(seed=0, rows=SCENE_ROWS, gap=SCENE_GAP):
    """A SimpleNamespace of arrays in metres: twelve players (slots 0-5 home, 6-11 away) and a ball that a player of one
    team or the other carries, half a metre from him, for stretches of 60 to 140 frames.

    The teams take the ball in turn, each for 60 to 140 frames, so that the classes of the teacher's rows are balanced.
    A uniform shift of everybody along the pitch follows the carrier's team, so that the shape of the teams says who
    has the ball (`adv_gap` is positive while home has it, and the teams swap ends in the second block). The truth is
    the carrier's team, flipped on 8% of the stretches (the teacher's noise). On 8% of the stretches the ball goes
    out: the carrier and the ball are put beyond the goal line, so that the dead-ball stump below marks them dead and
    the teacher's mask removes them; on 6% there is a stoppage with everybody where they were, which the truth calls
    dead and the model does not, so that the teacher has rows on frames whose truth is D. The ball is missing in long
    runs (about 40% of the frames), as ours is, and three times a block nobody is tracked for 25 frames (where P07 has
    no opinion and the teacher no row). Ungraded rows hold nothing.
    """
    rng = np.random.default_rng([27, seed])
    length, _ = SCENE_PITCH
    n = sum(rows) + gap
    graded = [(1, 0, rows[0]), (2, rows[0] + gap, n)]
    teams = np.array([0] * 6 + [1] * 6, dtype=np.int8)
    formation = np.concatenate([np.stack([np.linspace(15.0, 45.0, 6), np.linspace(10.0, 58.0, 6)], axis=1),
                                np.stack([np.linspace(60.0, 90.0, 6), np.linspace(58.0, 10.0, 6)], axis=1)])
    players, ball = np.full((n, SLOTS, 2), np.nan), np.full((n, 2), np.nan)
    truth_team, dead, inside = np.full(n, -1, dtype=np.int8), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    stretches = []
    for half, start, stop in graded:
        m = stop - start
        lengths = []
        while sum(lengths) < m:
            lengths.append(int(rng.integers(60, 141)))
        team_of = (np.arange(len(lengths)) + int(rng.integers(2))) % 2
        slot_of = rng.integers(6, size=len(lengths)) + 6 * team_of
        flip, outside_of = rng.random(len(lengths)) < 0.08, rng.random(len(lengths)) < 0.08
        stopped_of = (rng.random(len(lengths)) < 0.06) & ~outside_of
        carrier_team, carrier = np.repeat(team_of, lengths)[:m], np.repeat(slot_of, lengths)[:m]
        truth_team[start:stop] = np.repeat(np.where(flip, 1 - team_of, team_of), lengths)[:m]
        dead[start:stop], inside[start:stop] = np.repeat(outside_of | stopped_of, lengths)[:m], True
        outside = np.repeat(outside_of, lengths)[:m]
        edges = np.concatenate([[0], np.cumsum(lengths)])
        stretches += [(start + int(a), min(start + int(b), stop), int(t)) for a, b, t in zip(edges[:-1], edges[1:],
                                                                                              team_of)]
        base = formation if half == 1 else formation * [-1.0, 1.0] + [length, 0.0]
        lead = np.convolve(np.where(carrier_team == 0, 1.0, -1.0), np.ones(51) / 51, mode="same")
        xy = base[None] + np.cumsum(rng.normal(size=(m, SLOTS, 2)) * 0.03, axis=0)
        xy[:, :, 0] += ((1.0 if half == 1 else -1.0) * 6.0 * lead)[:, None]
        out = np.flatnonzero(outside)
        xy[out, carrier[out]] = [-3.0, 34.0]                      # the carrier and, half a metre on, the ball: out
        players[start:stop] = xy
        ball[start:stop] = xy[np.arange(m), carrier] + [0.5, 0.0]
    seen, position, run_is_seen = np.zeros(n, dtype=bool), 0, True
    while position < n:
        span = int(rng.integers(200, 601)) if run_is_seen else int(rng.integers(150, 401))
        seen[position:position + span] = run_is_seen
        position, run_is_seen = position + span, not run_is_seen
    ball[~seen] = np.nan
    tags = []
    for start, stop, team in stretches:
        if rng.random() < 0.65:                                   # a tag on two stretches in three, right on 85%
            named = team if rng.random() < 0.85 else 1 - team
            tags.append({"code": "BUILD UP", "start_frame": start, "end_frame": stop,
                         "tags": {"Team": "Barcelona" if named == 0 else "Atletic Club"}})
    for _, start, stop in graded:
        for hole in rng.integers(start + 100, stop - 100, size=3):
            players[hole:hole + 25] = np.nan                      # nobody tracked: P07 has nothing to read there
    return types.SimpleNamespace(players=players, ball=ball, teams=teams, graded=graded, truth_team=truth_team,
                                 dead=dead, inside=inside, tags=tags)


def scene_match(match_id, seed=0, mirror=False, rows=SCENE_ROWS):
    """The scene as a DflMatch (DFL's codes: possession 0 home, 1 away; status 1 alive, 0 dead); `mirror` swaps H, A."""
    scene = carry_scene(seed, rows)
    n, scale = len(scene.ball), np.array(SCENE_PITCH)
    possession = np.where(scene.inside, scene.truth_team, -1).astype(np.int8)
    if mirror:
        possession = np.where(possession >= 0, 1 - possession, possession).astype(np.int8)
    arrays = {"frame_number": np.where(scene.inside, np.arange(n), -1).astype(np.int64),
              "players_xy": scene.players / scale,
              "players_speed": np.where(np.isnan(scene.players[:, :, 0]), np.nan, 0.0),
              "player_team": scene.teams, "ball_xy": scene.ball / scale,
              "ball_z": np.where(scene.inside, 0.0, np.nan), "ball_speed": np.where(scene.inside, 0.0, np.nan),
              "ball_status": np.where(scene.inside, np.where(scene.dead, 0, 1), -1).astype(np.int8),
              "ball_possession": possession}
    for array in arrays.values():
        array.setflags(write=False)
    return DflMatch(match_id=match_id, pitch=SCENE_PITCH, fps=25, graded=tuple(scene.graded),
                    kickoffs={half: start for half, start, _ in scene.graded}, report={}, **arrays)


def scene_inputs(seed=5, rows=SCENE_ROWS):
    """BAR-ATH's stand-in: the scene's tracking as an Inputs-like object, with Smart Tagging's tags, as `tag_stats`
    builds its own. Its game is BAR-ATH, so that the source takes `full`.

    Twelve more slots are added, six for each team, in a corner far from the ball and present on some frames of every
    250: while the home ghosts are there home shows 12 slots and while the away ones are there away does, so that the
    over-eleven shares are not trivially 0 (a slot is a tracklet, and a team can show more than 11, T32)."""
    scene = carry_scene(seed, rows)
    n = len(scene.ball)
    ghosts = np.full((n, 12, 2), np.nan)
    for j in range(12):
        ghosts[:, j] = [3.0 + 0.5 * j, 3.0]
    phase = (np.arange(n) // 250) % 5
    ghosts[~scene.inside] = np.nan
    ghosts[:, :6][phase != 1] = np.nan                     # home ghosts on the second 250 of each 1,250 frames
    ghosts[:, 6:][phase != 3] = np.nan                     # away ghosts on the fourth
    teams = np.concatenate([scene.teams, np.array([0] * 6 + [1] * 6, dtype=np.int8)])
    inputs = play_inputs(np.concatenate([scene.players, ghosts], axis=1), scene.ball, teams=teams,
                         graded=scene.graded, pitch=SCENE_PITCH)
    inputs.game, inputs.n_frames, inputs.smart_tagging = "BAR-ATH", n, scene.tags
    return inputs


def shape_trees():
    """A shape model of three stumps on adv_gap: z is +-0.3 where the gap is within 3 m and +-0.8 beyond it (worked out
    in the symmetrised form (r(x) - r(swap(x))) / 2, which cancels the baseline), and 0 where the gap is NaN."""
    gap = SHAPE_COLUMN["adv_gap"]
    return {"baseline": 0.0, "prior_home": 0.5,
            "trees": [stump(gap, 0.0, -0.3, 0.3), stump(gap, 3.0, 0.0, 0.5), stump(gap, -3.0, -0.5, 0.0)]}


def synthetic_settings(root, ids=IDS, epsilon=0.07, min_prob=MIN_PROB):
    """Settings over two small model files in `root`: the shape stumps (a fold for each id, and `full`) and a dead-ball
    stump that calls the ball dead where it is more than half a metre beyond a line."""
    shape = write_shape_model(root, folds={match_id: shape_trees() for match_id in ids}, full=shape_trees(),
                              name="shape.json")
    dead = write_model(root, folds={match_id: touch_dead_model(3.0, -3.0) for match_id in ids},
                       full=touch_dead_model(3.0, -3.0), name="dead.json")
    return selftrain.Settings(teacher=dict(TEACHER), epsilon=epsilon, alive_model=dead, shape_model=shape,
                              min_prob=min_prob, c_team=0.76)


@functools.lru_cache(maxsize=1)
def fixture():
    """(folder, matches, settings, report): the gate (clean profile, one seed) on two synthetic matches, run once."""
    root = tempfile.mkdtemp(prefix="selftrain_check_")
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    matches = {match_id: scene_match(match_id, seed) for seed, match_id in enumerate(IDS)}
    settings = synthetic_settings(root)
    return root, matches, settings, selftrain.gate("clean", 1, matches=matches, settings=settings)


def quietly(function, *arguments):
    """(result, what it printed)."""
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        result = function(*arguments)
    return result, shown.getvalue()


# -- the settings -----------------------------------------------------------------------------------------------------

def config_text(drop=(), confidence=0.93, extra_touch="", shape_extra=""):
    entries = {
        "controlled_touch": 'radius = 2.0\nmax_rel_speed = 5.0\nmin_frames = 2\nconfidence = {}\nhold = 0.6\n{}'.format(
            confidence, extra_touch),
        "dead_ball_model": 'model = "pipeline/models/dead_ball_hgb.json"\n',
        "shape_possession": 'model = "pipeline/models/shape_possession_hgb.json"\nscope = "all"\n' + shape_extra,
        "smart_tagging": "c_team = 0.7\n"}
    parts = ['[[evidence]]\nkind = "{}"\n{}'.format(kind, body) for kind, body in entries.items() if kind not in drop]
    return "\n".join(parts) + '\n[decoder]\nkind = "majority_vote"\nhalf_width = 2\n'


def test_selftrain_settings_read_main():
    got = selftrain.settings()
    assert isinstance(got, selftrain.Settings)
    assert got._fields == ("teacher", "epsilon", "alive_model", "shape_model", "min_prob", "c_team")
    teacher = got.teacher
    assert (teacher["radius"], teacher["max_rel_speed"], teacher["min_frames"], teacher["duel_radius"]) == (
        1.5, 6.0, 1, 1.0)
    assert sorted(teacher) == ["duel_radius", "max_rel_speed", "min_frames", "radius"], "confidence, hold: the source's"
    assert got.epsilon == 0.07 and isinstance(got.epsilon, float), got.epsilon
    assert got.alive_model == "pipeline/models/dead_ball_hgb.json"
    assert got.shape_model == "pipeline/models/shape_possession_hgb.json"
    assert got.min_prob == 0.02 and got.c_team == 0.76
    assert selftrain.read_settings is selftrain.settings, "gate and bar_ath take a parameter of that name"
    # the defaults of an entry are the source's own: no duel_radius is 1.0 and no min_prob is shape_possession's 0.02
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "trial.toml", config_text(extra_touch="duel_radius = 0.8\n"))
        other = selftrain.settings(path)
        assert other.teacher == {"radius": 2.0, "max_rel_speed": 5.0, "min_frames": 2, "duel_radius": 0.8}, other
        assert other.epsilon == round(1.0 - 0.93, 12) == 0.07 and other.c_team == 0.7 and other.min_prob == 0.02
        assert selftrain.settings(write_text(root, "t2.toml", config_text(confidence=0.9))).epsilon == 0.1
        assert selftrain.settings(write_text(root, "t3.toml", config_text(shape_extra="min_prob = 0.05\n"))
                                  ).min_prob == 0.05
        assert selftrain.settings(write_text(root, "t4.toml", config_text())).teacher["duel_radius"] == 1.0
        # an entry that is not there is refused, naming its kind; so is one given twice, and a confidence of 1
        for kind in ("controlled_touch", "dead_ball_model", "shape_possession", "smart_tagging"):
            expect_error(ValueError, lambda: selftrain.settings(write_text(root, "no.toml", config_text(drop=(kind,)))),
                         "selftrain", kind)
        twice = config_text() + '\n[[evidence]]\nkind = "controlled_touch"\nname = "again"\nradius = 1.0\n' \
                                'max_rel_speed = 6.0\nmin_frames = 1\nconfidence = 0.9\n'
        expect_error(ValueError, lambda: selftrain.settings(write_text(root, "twice.toml", twice)), "controlled_touch",
                     "exactly one")
        expect_error(ValueError, lambda: selftrain.settings(write_text(root, "one.toml", config_text(confidence=1.0))),
                     "confidence", "epsilon")
        expect_error(ValueError, lambda: selftrain.settings(os.path.join(root, "absent.toml")), "absent.toml")


def paragraphs_after_description(config):
    """The blank-line separated paragraphs of a config's text after its description line, each stripped: what is left
    when the header comment and the description, each config's own, are taken off."""
    _, found, rest = config.text.partition("\ndescription = ")
    assert found, "{} has no description line".format(config.stem)
    return [paragraph.strip() for paragraph in rest.split("\n", 1)[1].split("\n\n")]


def test_selftrain_r027_configs_are_the_base_with_the_source_swapped():
    """R027's two configs are the base config, r024_both.toml, with its one `shape_possession` entry replaced by the
    declared `shape_selftrain` one and nothing else re-chosen; the second has no [[refine]], for the shift's
    re-derivation (R012's rule).

    The base is r024_both.toml and not main.toml, so that the check holds after an automatic adoption, when main.toml
    is itself the candidate and has no `shape_possession` entry to replace. Today the two have the same entries. Both
    configs load against the real registries. After the header comment and the description, whose texts are each
    config's own, r027_post's paragraphs are the base's with the `shape_possession` one replaced by R027_ENTRY, the
    declared entry, to the character, and r027_preshift's are r027_post's without the last, the refiner's. Read as
    configs, every other source, the decoder and the refiners are the base's, and the entry's teacher, epsilon, models,
    clip, scope and weight are what selftrain.settings(base) and the base's own entries give: the touch rule's table
    with its duel radius, 1 minus its confidence, the two model paths, P07's clip, scope and weight. So no constant is
    re-chosen in them. The header comment names R027 and the descriptions say what each config is.
    """
    base = load_config(BASE_CONFIG)
    post = load_config(os.path.join(R027_CONFIGS, "r027_post.toml"))
    preshift = load_config(os.path.join(R027_CONFIGS, "r027_preshift.toml"))
    expected = [R027_ENTRY if paragraph.startswith('[[evidence]]\nkind = "shape_possession"') else paragraph
                for paragraph in paragraphs_after_description(base)]
    assert expected.count(R027_ENTRY) == 1, "the base has exactly one shape_possession entry"
    assert paragraphs_after_description(post) == expected
    assert expected[-1].startswith("# ") and "[[refine]]" in expected[-1], "the last paragraph is the refiner's"
    assert paragraphs_after_description(preshift) == expected[:-1]
    assert [component.kind for component in post.refine] == ["shift_edges"] and preshift.refine == ()
    swapped = {"shape_possession": "shape_selftrain"}
    for config in (post, preshift):
        assert config.text.startswith("# R027"), "the header comment names R027: " + config.stem
        assert "R027" in config.description and "shape_selftrain" in config.description, config.description
        assert config.description != base.description and (config.seed, config.scope) == (base.seed, base.scope)
        assert [swapped.get(c.kind, c.kind) for c in base.evidence] == [c.kind for c in config.evidence]
        assert [c.name for c in config.evidence] == [swapped.get(c.name, c.name) for c in base.evidence]
        assert config.decoder == base.decoder
        for ours, theirs in zip(config.evidence, base.evidence):
            if theirs.kind not in swapped:
                assert ours == theirs, (ours, theirs)
    assert post.refine == base.refine and "shift_edges" in post.description
    assert "without shift_edges" in preshift.description, preshift.description
    old = [c for c in base.evidence if c.kind == "shape_possession"][0]
    touch = [c for c in base.evidence if c.kind == "controlled_touch"][0].params
    entry = [c for c in post.evidence if c.kind == "shape_selftrain"][0]
    assert entry == [c for c in preshift.evidence if c.kind == "shape_selftrain"][0]
    assert set(entry.params) == {"model", "alive_model", "teacher", "epsilon", "scope", "min_prob"}, entry.params
    settings = selftrain.settings(BASE_CONFIG)
    teacher = {key: touch[key] for key in ("radius", "max_rel_speed", "min_frames")}
    assert entry.params["teacher"] == settings.teacher == dict(teacher, duel_radius=touch.get("duel_radius", 1.0))
    assert entry.params["teacher"] == recalibration.check_teacher(entry.params["teacher"])
    assert entry.params["epsilon"] == settings.epsilon == round(1.0 - touch["confidence"], 12) == 0.07
    assert entry.params["alive_model"] == settings.alive_model and entry.params["model"] == settings.shape_model
    assert entry.params["min_prob"] == settings.min_prob == old.params["min_prob"]
    assert (entry.params["scope"], entry.params["model"], entry.weight) == ("all", old.params["model"], old.weight)
    assert entry.weight == 0.2 and entry.name == "shape_selftrain", entry


def test_selftrain_constants_are_the_designs():
    # Fixed before any run (the design's sections 2, 3 and 6): to change one is a decision, not an edit.
    assert (selftrain.GATE_MIN_GAIN, selftrain.GATE_MAX_HARM, selftrain.TAG_MARGIN) == (0.002, 0.002, 0.005)
    assert (selftrain.HEAD_PENALTY, selftrain.HEAD_TOLERANCE, selftrain.HEAD_MAX_STEPS) == (0.01, 1e-10, 50)
    assert selftrain.FOLDER == ("runs", "selftrain")
    assert selftrain._calibration_for is train_shape._calibration_for, "imported, not copied"
    assert selftrain._seed_list is train_shape._seed_list, "imported, not copied"
    parser = selftrain.build_parser()
    args = parser.parse_args(["gate"])
    assert (args.command, args.profile, args.seeds, args.matches, args.data_dir, args.no_write) == (
        "gate", "tracking", 3, None, None, False)
    args = parser.parse_args(["gate", "--profile", "context", "--seeds", "2", "--matches", "A", "B", "--data-dir", "D",
                              "--no-write"])
    assert (args.profile, args.seeds, args.matches, args.data_dir, args.no_write) == (
        "context", 2, ["A", "B"], "D", True)
    args = parser.parse_args(["bar-ath", "--gate-report", "g.json", "--data-dir", "D", "--no-write"])
    assert (args.command, args.gate_report, args.data_dir, args.no_write) == ("bar-ath", "g.json", "D", True)
    for bad in ([], ["bar-ath"], ["gate", "--seeds", "0"], ["nope"], ["gate", "--gate-report", "g.json"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                selftrain.main(bad)
            except SystemExit as stop:
                assert stop.code == 2, (bad, stop.code)
            else:
                raise AssertionError("{} was accepted".format(bad))


# -- rung 3's head ----------------------------------------------------------------------------------------------------

def head_sample(n=3000, p=8, seed=0):
    """(z, y, X): a model's logit, noisy labels that follow it and two of the features, and features with NaN holes, one
    constant column and one column that is all NaN."""
    rng = np.random.default_rng([28, seed])
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 4.0, size=p) + rng.normal(size=p) * 3.0
    z = rng.normal(size=n) * 1.5
    first, second = [(X[:, j] - X[:, j].mean()) / X[:, j].std() for j in (0, 1)]
    score = 0.8 * z + 0.6 * first - 0.4 * second
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64)
    X[:, 3] = 7.25
    X[:, 6] = np.nan
    X[rng.random(X.shape) < 0.05] = np.nan
    X[:, 6] = np.nan
    return z, y, X


def reference_standardised(X):
    """X standardised the declared way, by plain loops: the mean and the population SD of the finite entries of a
    column, NaN to 0 afterwards, a column with no spread to 0."""
    out = np.zeros(X.shape)
    for j in range(X.shape[1]):
        values = [value for value in X[:, j] if not math.isnan(value)]
        if not values or max(values) == min(values):
            continue
        mean = math.fsum(values) / len(values)
        sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
        out[:, j] = [0.0 if math.isnan(value) else (value - mean) / sd for value in X[:, j]]
    return out


def scipy_head(z, y, X, penalty=0.01):
    """(theta, Xs): scipy's L-BFGS-B minimum of the design's objective, theta = (intercept, beta), and the features it
    was fitted on (standardised as declared): sum w * log-loss + (penalty / 2) |beta|^2, each class weighing a half."""
    from scipy.optimize import minimize
    Xs = reference_standardised(X)
    n1 = int(y.sum())
    w = np.where(y == 1, 0.5 / n1, 0.5 / (len(y) - n1))

    def objective(theta):
        t = z + theta[0] + Xs @ theta[1:]
        loss = np.where(y == 1, np.logaddexp(0.0, -t), np.logaddexp(0.0, t))
        return float(np.sum(w * loss) + 0.5 * penalty * (theta[1:] @ theta[1:]))

    def gradient(theta):
        t = z + theta[0] + Xs @ theta[1:]
        residual = w * (1.0 / (1.0 + np.exp(-t)) - y)
        return np.concatenate([[residual.sum()], Xs.T @ residual + penalty * theta[1:]])

    solved = minimize(objective, np.zeros(Xs.shape[1] + 1), jac=gradient, method="L-BFGS-B", tol=1e-13,
                      options={"maxiter": 20000, "maxfun": 200000})
    return solved.x, Xs


def test_selftrain_head_fit_matches_scipy():
    z, y, X = head_sample()
    epsilon = 0.07
    head = selftrain.head_fit(z, y, X, epsilon=epsilon)
    theta, Xs = scipy_head(z, y, X)
    solved = types.SimpleNamespace(x=theta)
    worst = float(np.abs(head.beta - solved.x[1:]).max())
    print("      head_fit against L-BFGS-B: max |beta difference| {:.2e}, steps {}".format(worst, head.steps))
    assert worst <= 1e-6, worst
    assert np.abs(head.beta).max() > 0.2, "the fit must find something for this to mean much"
    assert head.beta[3] == 0.0 and head.beta[6] == 0.0, "a constant column and an all-NaN column carry nothing"
    assert 3 <= head.steps <= selftrain.HEAD_MAX_STEPS
    # the intercept is E2 on z plus the head, and not the balanced fit's own
    expected = recalibration.noise_aware_offset(z + Xs @ head.beta, y, epsilon=epsilon).offset
    assert abs(head.offset - expected) <= 2e-9, (head.offset, expected)
    assert head.offset != solved.x[0], "re-fitted by E2, so that it carries no pull towards the minority class"
    standardised = selftrain.standardised(X, head.mean, head.scale)
    assert np.abs(standardised - Xs).max() <= 1e-12
    assert head.offset == recalibration.noise_aware_offset(z + selftrain.head_score(head, X), y, epsilon=epsilon).offset
    # applied to rows: z + x~ beta + offset, with NaN z staying NaN and NaN features adding nothing
    holey = z.copy()
    holey[[5, 50]] = np.nan
    applied = selftrain.head_logit(head, holey, X)
    assert np.array_equal(np.isnan(applied), np.isnan(holey))
    assert np.abs(applied[~np.isnan(holey)] - (z + Xs @ head.beta + head.offset)[~np.isnan(holey)]).max() <= 1e-12
    # every column constant: beta is exactly 0 and the offset is E2's on z
    for constant in (np.full((len(z), 5), 3.0), np.full((len(z), 5), np.nan), np.zeros((len(z), 1))):
        flat = selftrain.head_fit(z, y, constant, epsilon=epsilon)
        assert not np.any(flat.beta), flat.beta
        assert flat.offset == recalibration.noise_aware_offset(z, y, epsilon=epsilon).offset
        assert np.array_equal(selftrain.head_logit(flat, z, constant), z + flat.offset)


def test_selftrain_head_is_standardised_and_exact():
    # the standardisation of the design's section 3, on a column worked out by hand: mean 2, population SD sqrt(2/3)
    X = np.array([[1.0, 5.0, np.nan], [2.0, 5.0, np.nan], [3.0, 5.0, np.nan], [np.nan, 5.0, np.nan]])
    z, y = np.array([0.1, -0.2, 0.3, 0.0]), np.array([1, 0, 1, 0])
    mean, scale = selftrain.standardisation(X)
    assert mean[0] == 2.0 and abs(scale[0] - math.sqrt(2.0 / 3.0)) <= 1e-15
    assert scale[1] == 0.0 and scale[2] == 0.0, "no spread: scale 0, and the column becomes 0"
    Xs = selftrain.standardised(X, mean, scale)
    assert abs(Xs[0, 0] + 1.0 / math.sqrt(2.0 / 3.0)) <= 1e-15 and Xs[1, 0] == 0.0 and Xs[3, 0] == 0.0, Xs
    assert not Xs[:, 1:].any() and not np.isnan(Xs).any()
    # a column that is constant up to rounding of its mean is still constant: 0.1 three times has a mean that is not 0.1
    third = np.full((3, 1), 0.1)
    assert selftrain.standardisation(third)[1][0] == 0.0
    # the same bits whatever the order of the rows, in every field
    z, y, X = head_sample(n=1500, seed=3)
    base = selftrain.head_fit(z, y, X, epsilon=0.07)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(len(z))
        again = selftrain.head_fit(z[order], y[order], X[order], epsilon=0.07)
        assert again.offset == base.offset and again.steps == base.steps
        for name in ("beta", "mean", "scale"):
            assert np.array_equal(getattr(again, name), getattr(base, name)), (seed, name)
    # and on one thread as on four: a matrix product would not promise it, the exact sums do
    from threadpoolctl import threadpool_limits
    for limit in (1, 4):
        with threadpool_limits(limits=limit):
            again = selftrain.head_fit(z, y, X, epsilon=0.07)
        assert again.offset == base.offset and all(np.array_equal(getattr(again, name), getattr(base, name))
                                                   for name in ("beta", "mean", "scale")), limit
    # its sums are exact ones: math.fsum over the rows, a Hessian and a gradient at every step
    calls, real = [], math.fsum
    math.fsum = lambda values: calls.append(len(values)) or real(values)
    try:
        selftrain.head_fit(z, y, X, epsilon=0.07)
    finally:
        math.fsum = real
    assert len(calls) >= 200 and set(calls) >= {len(z)}, "the Newton sums are not math.fsum: {}".format(calls[:3])
    # what it refuses, with a sentence that names the fix
    expect_error(ValueError, lambda: selftrain.head_fit(z, np.ones(len(z), dtype=np.int64), X, epsilon=0.07),
                 "selftrain", "both classes")
    expect_error(ValueError, lambda: selftrain.head_fit(z[:-1], y, X, epsilon=0.07), "selftrain", "rows")
    expect_error(ValueError, lambda: selftrain.head_fit(z, y, X[:, 0], epsilon=0.07), "selftrain", "2-D")
    expect_error(ValueError, lambda: selftrain.head_fit(np.where(np.arange(len(z)) == 3, np.nan, z), y, X,
                                                        epsilon=0.07), "selftrain", "finite")
    expect_error(ValueError, lambda: selftrain.head_fit(z, y, np.where(np.arange(len(z))[:, None] == 3, np.inf, X),
                                                        epsilon=0.07), "selftrain", "infinite")
    expect_error(ValueError, lambda: selftrain.head_fit(z, np.where(np.arange(len(z)) == 3, 2, y), X, epsilon=0.07),
                 "selftrain", "0 or 1")
    for bad in (0.0, -0.01, True, "0.01", float("nan")):
        expect_error(ValueError, lambda: selftrain.head_fit(z, y, X, epsilon=0.07, penalty=bad), "selftrain", "penalty")
    expect_error(ValueError, lambda: selftrain.head_fit(z, y, X, epsilon=0.7), "recalibration", "epsilon")
    # no convergence in the steps allowed is an error of its own, not a wrong answer
    real_steps = selftrain.HEAD_MAX_STEPS
    selftrain.HEAD_MAX_STEPS = 2
    try:
        error = expect_error(selftrain.NoConvergence, lambda: selftrain.head_fit(z, y, X, epsilon=0.07), "selftrain",
                             "2 steps")
    finally:
        selftrain.HEAD_MAX_STEPS = real_steps
    assert issubclass(selftrain.NoConvergence, ValueError) and "penalty" in error
    expect_error(ValueError, lambda: selftrain.head_logit(base, z, X[:, :3]), "selftrain", "columns")
    expect_error(ValueError, lambda: selftrain.head_logit(base, z[:-1], X), "selftrain", "rows")


def hard_design(seed, n=1500, p=26):
    """(z, y, X): 26 features that are noisy copies of six signals (some nearly exact copies, as `adv_gap` is of
    `h_adv_mean` and `a_adv_mean` in the real ones), with NaN holes, and labels that may all but follow the first
    signal.
    Plain Newton overshoots on a good share of them."""
    rng = np.random.default_rng([31, seed])
    base = rng.normal(size=(n, 6))
    X = np.concatenate([base[:, [j % 6]] * rng.uniform(0.5, 3.0)
                        + rng.normal(size=(n, 1)) * rng.choice([1e-3, 0.05, 1.0]) + rng.normal() * 30.0
                        for j in range(p)], axis=1)
    X[rng.random(X.shape) < rng.choice([0.0, 0.1, 0.5])] = np.nan
    z = rng.normal(size=n) * rng.choice([0.5, 2.0, 5.0]) + rng.normal() * rng.choice([0.0, 1.0, 3.0])
    score = rng.choice([0.0, 1.0, 5.0, 40.0]) * np.nan_to_num(base[:, 0]) + 0.5 * z + rng.normal() * 2.0
    return z, (rng.random(n) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64), X


def test_selftrain_head_survives_a_design_that_plain_newton_cannot():
    # Plain Newton overshoots on these designs: its Hessian goes singular (every sigma(1 - sigma) rounds to 0, and
    # nothing penalises the intercept) and it reports no convergence, which would read as "rung 3 does nothing here".
    # With the step halved while it does not lower the objective, it reaches the same minimum as an optimiser that never
    # overshoots, in a handful of steps.
    real = selftrain.HEAD_LINE_SEARCH
    for seed in (1, 38, 47):
        z, y, X = hard_design(seed)
        selftrain.HEAD_LINE_SEARCH = math.inf
        try:
            expect_error(selftrain.NoConvergence, lambda: selftrain.head_fit(z, y, X, epsilon=0.07), "selftrain",
                         "Newton")
        finally:
            selftrain.HEAD_LINE_SEARCH = real
        head = selftrain.head_fit(z, y, X, epsilon=0.07)
        theta, _ = scipy_head(z, y, X)
        worst = float(np.abs(head.beta - theta[1:]).max())
        print("      hard design {}: {} steps, max |beta - L-BFGS-B| {:.2e}".format(seed, head.steps, worst))
        assert head.steps <= 15 and worst <= 5e-6, (seed, head.steps, worst)
    # near the optimum the steps are Newton's own: switching the safeguard off changes nothing where nothing overshoots
    z, y, X = head_sample()
    kept = selftrain.head_fit(z, y, X, epsilon=0.07)
    selftrain.HEAD_LINE_SEARCH = math.inf
    try:
        plain = selftrain.head_fit(z, y, X, epsilon=0.07)
    finally:
        selftrain.HEAD_LINE_SEARCH = real
    assert np.array_equal(kept.beta, plain.beta) and kept.offset == plain.offset and kept.steps == plain.steps


# -- the oracles' arithmetic and the decision -------------------------------------------------------------------------

def hand_cell(b, b_gap, b_touch):
    return {"seed": 0, "half": 1, "b": {"rung1": b}, "b_gap": b_gap, "b_touch": b_touch}


def test_selftrain_oracle_arithmetic_by_hand():
    # match one, cells (b, b*_gap, b*_touch): |b*_gap| is 0.5, 0.2 and 0.4 (the cell with no gate oracle is left out of
    # everything, and of the band); |b - b*_gap| is 0.1, 0.1 and 0.4; |b - b*_touch| is 0.2, 0.2 and 0.4;
    # |b*_touch - b*_gap| is 0.1, 0.3 and 0.8
    cells_one = [hand_cell(0.4, 0.5, 0.6), hand_cell(-0.1, -0.2, 0.1), hand_cell(0.0, 0.4, -0.4),
                 hand_cell(9.0, None, 0.0)]
    cells_two = [hand_cell(0.0, 0.1, None), hand_cell(0.2, 0.2, 0.5)]
    one, two = selftrain.oracle_means(cells_one), selftrain.oracle_means(cells_two)
    near = lambda a, b: abs(a - b) <= 1e-12                                              # noqa: E731
    assert (one["cells"], one["cells_gap"], one["cells_both"]) == (4, 3, 3), one
    assert near(one["B"], (0.5 + 0.2 + 0.4) / 3) and near(one["E"], (0.1 + 0.1 + 0.4) / 3), one
    assert near(one["noise"], (0.2 + 0.2 + 0.4) / 3) and near(one["transfer"], (0.1 + 0.3 + 0.8) / 3), one
    assert one["band"] == 0.5
    # match two: the cell with no touch oracle still counts for B and E, and not for the decomposition
    assert (two["cells"], two["cells_gap"], two["cells_both"]) == (2, 2, 1), two
    assert near(two["B"], 0.15) and near(two["E"], 0.05) and near(two["noise"], 0.3) and near(two["transfer"], 0.3)
    assert two["band"] == 0.2
    # the means of matches, each match once (a match with many cells does not weigh more), and the largest of the bands
    total = selftrain.oracle_summary({"A": one, "B": two})
    assert near(total["B"], ((0.5 + 0.2 + 0.4) / 3 + 0.15) / 2) and near(total["E"], (0.2 + 0.05) / 2), total
    assert near(total["E_over_B"], total["E"] / total["B"]) and total["band"] == 0.5
    assert near(total["noise"], ((0.2 + 0.2 + 0.4) / 3 + 0.3) / 2) and near(total["transfer"], (1.2 / 3 + 0.3) / 2)
    assert total["matches"] == 2
    # nothing defined: None for every mean, and no band; a match with no oracle at all is left out of the means
    empty = selftrain.oracle_means([hand_cell(0.0, None, None)])
    assert (empty["cells"], empty["cells_gap"], empty["B"], empty["E"], empty["band"]) == (1, 0, None, None, None)
    assert selftrain.oracle_summary({"A": empty}) == {"matches": 0, "B": None, "E": None, "E_over_B": None,
                                                      "noise": None, "transfer": None, "band": None}
    assert near(selftrain.oracle_summary({"A": one, "B": empty})["B"], (0.5 + 0.2 + 0.4) / 3)
    zero = selftrain.oracle_summary({"A": selftrain.oracle_means([hand_cell(0.0, 0.0, 0.0)])})
    assert zero["B"] == 0.0 and zero["E_over_B"] is None, "E/B of a zero B is not a number"


def variants_of(base, rung1, rung3):
    return {"variants": {"base": {"log_loss": base}, "rung1": {"log_loss": rung1}, "rung3": {"log_loss": rung3}}}


def test_selftrain_gate_verdict_at_the_edges():
    def verdict(pooled, per_match):
        return selftrain.gate_verdict({name: variants_of(*losses) for name, losses in per_match.items()},
                                      variants_of(*pooled))
    ok = {"M1": (0.60, 0.598, 0.60), "M2": (0.62, 0.62, 0.62)}
    assert verdict((0.61, 0.608, 0.61), ok) == (True, False), "a gain of exactly 0.002 and no harm: a go"
    assert verdict((0.61, 0.6081, 0.61), ok) == (False, False), "0.0019 is short of it, by however little"
    assert verdict((0.61, 0.605, 0.60), ok) == (True, True)
    assert verdict((0.61, 0.61, 0.60), ok) == (False, True)
    harm = dict(ok, M2=(0.62, 0.622, 0.62))
    assert verdict((0.61, 0.60, 0.61), harm) == (True, False), "a harm of exactly 0.002 is allowed"
    assert verdict((0.61, 0.60, 0.61), dict(ok, M2=(0.62, 0.6221, 0.62))) == (False, False), "0.0021 is a harm too many"
    assert verdict((0.61, 0.60, 0.61), dict(ok, M2=(0.62, 0.62, 0.6221))) == (True, False), "rung 3's harm is its own"
    assert verdict((0.61, 0.60, 0.59), dict(ok, M2=(0.62, 0.62, 0.6221))) == (True, False)
    assert verdict((0.61, 0.60, 0.59), ok) == (True, True)
    conditions = selftrain.gate_conditions({name: variants_of(*losses) for name, losses in harm.items()},
                                           variants_of(0.61, 0.60, 0.61), "rung1")
    assert conditions["go"] and abs(conditions["pooled_gain"] - 0.01) <= 1e-12 and conditions["gain_ok"]
    assert conditions["worst_match"] == "M2" and abs(conditions["worst_harm"] - 0.002) <= 1e-12
    assert conditions["harm_ok"] and sorted(conditions["per_match_harm"]) == ["M1", "M2"]
    assert abs(conditions["per_match_harm"]["M1"] + 0.002) <= 1e-12
    assert (conditions["min_gain"], conditions["max_harm"]) == (0.002, 0.002)
    # the exit codes: 0 for a go of rung 1, 4 for rung 3 alone, 1 for neither
    assert [selftrain.gate_exit_code(*pair) for pair in ((True, True), (True, False), (False, True), (False, False))] \
        == [0, 0, 4, 1]
    # a log-loss that is missing cannot be compared: an error, not a quiet no-go
    expect_error(ValueError, lambda: selftrain.gate_verdict({"M1": variants_of(None, 0.6, 0.6)},
                                                            variants_of(0.6, 0.6, 0.6)),
                 "selftrain", "log-loss")
    expect_error(ValueError, lambda: selftrain.gate_conditions({"M1": variants_of(0.6, 0.6, 0.6)},
                                                               variants_of(0.6, 0.6, 0.6), "rung2"),
                 "selftrain", "rung2")


# -- the gate on synthetic matches ------------------------------------------------------------------------------------

def test_selftrain_gate_on_synthetic_matches():
    root, matches, settings, report = fixture()
    assert set(report) == TOP_KEYS, sorted(set(report) ^ TOP_KEYS)
    assert report["kind"] == "selftrain_gate" and report["profile"] == {"name": "clean", "version": 1}
    assert report["seeds"] == [0] and report["matches"] == list(IDS) and report["calibration"] is None
    assert sorted(report["per_match"]) == list(IDS)
    for match_id in IDS:
        cell = report["per_match"][match_id]
        assert set(cell) == MATCH_KEYS, sorted(set(cell) ^ MATCH_KEYS)
        assert list(cell["variants"]) == list(VARIANTS) and sorted(cell["hold"]) == ["base", "rung1"]
        assert len(cell["cells"]) == 2, "one seed and two blocks"
        assert all(entry["fallback"]["rung1"] is None for entry in cell["cells"]), "the synthetic scene feeds the fits"
        assert any(abs(entry["b"]["rung1"]) > 1e-3 for entry in cell["cells"]), "an offset that is 0 proves little"
    assert set(report["pooled"]) == {"variants", "hold", "touch", "teacher", "fallbacks"}
    assert set(report["oracles"]) == {"matches", "B", "E", "E_over_B", "noise", "transfer", "band"}
    assert report["oracles"]["matches"] == 2 and report["oracles"]["band"] > 0
    assert set(report["gate"]) == {"rung1", "rung3", "exit_code", "verdict"}
    assert report["settings"]["epsilon"] == 0.07 and report["settings"]["teacher"] == TEACHER
    assert report["constants"]["min_rows"] == recalibration.MIN_ROWS == 500
    # the base variant is shape_possession's own home, on the gate's frames
    for match_id, match in matches.items():
        inputs = served_inputs(match, "clean", 0, None, None)
        home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=settings.shape_model, scope="all",
                                        min_prob=settings.min_prob).home
        codes, _ = truth.truth_codes(match)
        present = inputs.atd.ball_present
        frames = [f for f in range(len(codes)) if codes[f] in (H, A) and not present[f] and not math.isnan(home[f])]
        loss = math.fsum(-math.log(home[f] if codes[f] == H else 1.0 - home[f]) for f in frames) / len(frames)
        right = sum((home[f] >= 0.5) == (codes[f] == H) for f in frames) / len(frames)
        base = report["per_match"][match_id]["variants"]["base"]
        assert base["frames"] == len(frames) > 1000, (match_id, base["frames"])
        assert abs(base["log_loss"] - loss) <= 1e-12, (match_id, base["log_loss"], loss)
        assert abs(base["accuracy"] - right) <= 1e-12
        assert (home[frames] == settings.min_prob).any() or (home[frames] == 1.0 - settings.min_prob).any(), \
            "the clip must bind somewhere, or the tallies are not tested through it"
    # the pool is by frames, never a mean of the matches' means
    for name in VARIANTS:
        frames = [report["per_match"][m]["variants"][name]["frames"] for m in IDS]
        losses = [report["per_match"][m]["variants"][name]["log_loss"] for m in IDS]
        weighted = sum(f * loss for f, loss in zip(frames, losses)) / sum(frames)
        assert report["pooled"]["variants"][name]["frames"] == sum(frames)
        assert abs(report["pooled"]["variants"][name]["log_loss"] - weighted) <= 1e-12, name
    # the oracle figures are the arithmetic of the cells: a match's from its own, and the report's from the matches'
    for match_id in IDS:
        assert report["per_match"][match_id]["oracle"] == selftrain.oracle_means(report["per_match"][match_id]["cells"])
    assert report["oracles"] == selftrain.oracle_summary({m: report["per_match"][m]["oracle"] for m in IDS})
    assert report["oracles"]["band"] == max(abs(cell["b_gap"]) for m in IDS for cell in report["per_match"][m]["cells"])
    # the gate's section is the decision function's answer on the report's own numbers
    rung1_go, rung3_go = selftrain.gate_verdict(report["per_match"], report["pooled"])
    assert (report["gate"]["rung1"]["go"], report["gate"]["rung3"]["go"]) == (rung1_go, rung3_go)
    assert report["gate"]["exit_code"] == selftrain.gate_exit_code(rung1_go, rung3_go)
    assert report["gate"]["rung1"] == selftrain.gate_conditions(report["per_match"], report["pooled"], "rung1")
    # rung 1 with every offset forced to 0 is the base, exactly (E1 is not forced, so it is not: the check has teeth)
    real = recalibration.noise_aware_offset
    recalibration.noise_aware_offset = lambda z, y, *, epsilon: recalibration.Offset(0.0, 0.0, 0.5, 0.5)
    try:
        zero = selftrain.gate("clean", 1, matches={"RND001": matches["RND001"]}, settings=settings)
    finally:
        recalibration.noise_aware_offset = real
    variants = zero["per_match"]["RND001"]["variants"]
    assert variants["rung1"] == variants["base"] and variants["single"] == variants["base"], variants
    assert variants["e1"] != variants["base"]
    assert all(entry["b"]["rung1"] == 0.0 and entry["b"]["single"] == 0.0
               for entry in zero["per_match"]["RND001"]["cells"])
    assert report["per_match"]["RND001"]["variants"]["rung1"] != report["per_match"]["RND001"]["variants"]["base"]
    # a profile other than clean is served through its Calibration, and the seeds are counted
    small = {"RND001": scene_match("RND001", 4)}
    gapped = selftrain.gate("gaps", 2, matches=small, settings=settings, calibration=synthetic_calibration())
    assert gapped["profile"]["name"] == "gaps" and gapped["seeds"] == [0, 1]
    assert len(gapped["per_match"]["RND001"]["cells"]) == 4
    assert gapped["calibration"]["sha256"] and set(gapped["per_match"]["RND001"]["fallbacks"]) == {
        "rung1", "e1", "single", "rung3"}


def test_selftrain_gate_equals_plain_loops():
    root, matches, settings, report = fixture()
    epsilon, min_prob, floor = settings.epsilon, settings.min_prob, recalibration.MIN_ROWS
    pooled_tallies = {name: [0, 0.0, 0] for name in VARIANTS}
    silent = {"gaps": 0, "seen": 0}
    for match_id, match in matches.items():
        inputs = served_inputs(match, "clean", 0, None, None)
        run_context = RunContext(inputs=inputs, seed=0)
        z = shape_possession._logit_home(inputs, settings.shape_model, match_id,
                                         shape_possession.load(settings.shape_model)["folds"][match_id])
        y, rows = recalibration.teacher_labels(run_context, alive_model=settings.alive_model, teacher=settings.teacher)
        X = shape_features.features(inputs)[1]
        codes, _ = truth.truth_codes(match)
        present = inputs.atd.ball_present
        # alive frames where nobody is tracked: P07 has no opinion there, so they are in no stratum and no row (a gate
        # that counted them would be scoring a NaN, and that there are some is what gives the loops below their teeth)
        for f in range(len(z)):
            if codes[f] in (H, A) and math.isnan(z[f]):
                silent["seen" if present[f] else "gaps"] += 1
                assert math.isnan(y[f]), "nobody is present, so nobody is next to the ball"
        episode = controlled_touch.compute(run_context, **conventions.TOUCH).home
        blocks = [(start, stop) for _, start, stop in inputs.graded]
        usable = [bool(rows[f]) and math.isfinite(z[f]) for f in range(len(z))]

        def pool(skip):
            return [f for k, (start, stop) in enumerate(blocks) if k != skip for f in range(start, stop) if usable[f]]

        def enough(frames):
            homes = sum(int(y[f]) for f in frames)
            return homes >= floor and len(frames) - homes >= floor

        def fit(frames, how):
            if not enough(frames):
                return 0.0
            labels = np.array([int(y[f]) for f in frames])
            return how(np.array([z[f] for f in frames]), labels)

        e2 = lambda zs, labels: recalibration.noise_aware_offset(zs, labels, epsilon=epsilon).offset      # noqa: E731
        e1 = lambda zs, labels: recalibration.balanced_offset(zs, labels).offset                          # noqa: E731
        single = fit(pool(None), e2)

        def served(logit):
            p = 1.0 / (1.0 + math.exp(-logit))
            return min(max(p, min_prob), 1.0 - min_prob)

        def tally(frames, logit_of):
            loss = right = 0
            for f in frames:
                p, home = served(logit_of(f)), codes[f] == H
                loss += -math.log(p if home else 1.0 - p)
                right += (p >= 0.5) == home
            return [len(frames), loss, right]

        sums = {name: [0, 0.0, 0] for name in VARIANTS}
        hold_sums, touch = {name: [0, 0.0, 0] for name in ("base", "rung1")}, [0, 0, 0]
        cells = report["per_match"][match_id]["cells"]
        for k, (start, stop) in enumerate(blocks):
            block = range(start, stop)
            gate_frames = [f for f in block if codes[f] in (H, A) and not present[f] and math.isfinite(z[f])]
            hold_frames = [f for f in block if codes[f] in (H, A) and present[f] and math.isfinite(z[f])
                           and math.isnan(y[f])]
            offsets = {"rung1": fit(pool(k), e2), "e1": fit(pool(k), e1), "single": single}
            pool_k = pool(k)
            head = None
            if enough(pool_k):
                head = selftrain.head_fit(z[pool_k], np.array([int(y[f]) for f in pool_k]), X[pool_k], epsilon=epsilon)
            rung3 = selftrain.head_logit(head, z[start:stop], X[start:stop]) if head else z[start:stop]
            logit_of = {"base": lambda f: z[f], "rung1": lambda f: z[f] + offsets["rung1"],
                        "e1": lambda f: z[f] + offsets["e1"], "single": lambda f: z[f] + offsets["single"],
                        "rung3": lambda f: rung3[f - start]}
            for name in VARIANTS:
                sums[name] = [a + b for a, b in zip(sums[name], tally(gate_frames, logit_of[name]))]
            for name in ("base", "rung1"):
                hold_sums[name] = [a + b for a, b in zip(hold_sums[name], tally(hold_frames, logit_of[name]))]
            kept = [f for f in block if not math.isnan(episode[f]) and math.isfinite(z[f])]
            touch[0] += len(kept)
            for slot, name in ((1, "base"), (2, "rung1")):
                touch[slot] += sum((served(logit_of[name](f)) >= 0.5) == (episode[f] >= 0.5) for f in kept)
            # the cell: the offsets, the oracles, the fallbacks
            cell = cells[k]
            assert (cell["seed"], cell["half"]) == (0, k + 1)
            for name in ("rung1", "e1", "single"):
                assert cell["b"][name] == offsets[name], (match_id, k, name, cell["b"][name], offsets[name])
            assert abs(cell["b"]["rung3"] - (head.offset if head else 0.0)) <= 1e-12
            in_pool = [f for f in pool_k if codes[f] in (H, A)]
            b_gap = recalibration.balanced_offset(np.array([z[f] for f in gate_frames]),
                                                  np.array([int(codes[f] == H) for f in gate_frames])).offset
            b_touch = recalibration.balanced_offset(np.array([z[f] for f in in_pool]),
                                                    np.array([int(codes[f] == H) for f in in_pool])).offset
            assert cell["b_gap"] == b_gap and cell["b_touch"] == b_touch, (match_id, k, cell, b_gap, b_touch)
            assert (cell["rows_home"], cell["rows_away"]) == (sum(int(y[f]) for f in pool_k),
                                                              len(pool_k) - sum(int(y[f]) for f in pool_k))
            assert cell["fallback"] == {"rung1": None, "e1": None, "single": None, "rung3": None}
        cell = report["per_match"][match_id]
        for name in VARIANTS:
            frames, loss, right = sums[name]
            got = cell["variants"][name]
            assert got["frames"] == frames and abs(got["log_loss"] - loss / frames) <= 1e-9, (match_id, name, got)
            assert abs(got["accuracy"] - right / frames) <= 1e-12, (match_id, name)
            pooled_tallies[name] = [a + b for a, b in zip(pooled_tallies[name], sums[name])]
        for name in ("base", "rung1"):
            frames, loss, right = hold_sums[name]
            got = cell["hold"][name]
            assert frames > 100 and got["frames"] == frames, (name, got)
            assert abs(got["log_loss"] - loss / frames) <= 1e-9, (name, got)
            assert abs(got["accuracy"] - right / frames) <= 1e-12
        assert cell["touch"]["frames"] == touch[0] > 500, (match_id, cell["touch"])
        assert cell["touch"]["base"] == touch[1] / touch[0] and cell["touch"]["rung1"] == touch[2] / touch[0]
        # the teacher's figures, block by block, summed over the blocks
        known = sum(int(not math.isnan(y[f])) for f in range(len(y)))
        masked = sum(int(not math.isnan(y[f]) and not rows[f]) for f in range(len(y)))
        usable_rows = [f for f in range(len(y)) if usable[f]]
        alive_rows = [f for f in usable_rows if codes[f] in (H, A)]
        teacher = cell["teacher"]
        assert teacher["known_frames"] == known and masked > 0 and teacher["masked_share"] == masked / known
        assert teacher["rows"] == len(usable_rows) and teacher["rows_home"] == sum(int(y[f]) for f in usable_rows)
        assert teacher["rows_away"] == len(usable_rows) - teacher["rows_home"]
        assert teacher["precision_frames"] == len(alive_rows)
        assert teacher["precision"] == sum((y[f] == 1) == (codes[f] == H) for f in alive_rows) / len(alive_rows)
        assert 0.7 < teacher["precision"] < 1.0 and teacher["rows_truth_dead"] == len(usable_rows) - len(alive_rows) > 0
    assert silent["gaps"] > 0 and silent["seen"] > 0, silent
    for name in VARIANTS:
        assert report["pooled"]["variants"][name]["frames"] == pooled_tallies[name][0]
        pooled_loss = report["pooled"]["variants"][name]["log_loss"]
        assert abs(pooled_loss - pooled_tallies[name][1] / pooled_tallies[name][0]) <= 1e-9


def test_selftrain_gate_refuses_what_would_score_in_sample():
    _, matches, settings, _ = fixture()

    class Trap(dict):
        def __getitem__(self, key):
            raise AssertionError("match {} was loaded before the gate checked its models".format(key))

    trap = Trap(matches)
    with tempfile.TemporaryDirectory() as root:
        # a fold that is missing would fall back to `full`, which saw the match: refused, naming it, before a load
        short = synthetic_settings(root, ids=("RND001",))
        expect_error(BenchmarkError, lambda: selftrain.gate("clean", 1, matches=trap, settings=short), "selftrain",
                     "RND002", "fold", "in sample", "train_shape.py")
        # the dead-ball model's fold too: the teacher's mask would otherwise have seen the match's own flags
        shape = write_shape_model(root, folds={match_id: shape_trees() for match_id in IDS}, full=shape_trees(),
                                  name="whole.json")
        dead = write_model(root, folds={"RND001": touch_dead_model(3.0, -3.0)}, full=touch_dead_model(3.0, -3.0),
                           name="short_dead.json")
        expect_error(BenchmarkError, lambda: selftrain.gate("clean", 1, matches=trap, settings=settings._replace(
            shape_model=shape, alive_model=dead)), "dead_ball_model", "RND002", "fold", "train_dead_ball.py")
    for bad in (0, True, 1.5, "3"):
        expect_error(ValueError, lambda: selftrain.gate("clean", bad, matches=trap, settings=settings), "selftrain",
                     "seeds")
    expect_error(BenchmarkError, lambda: selftrain.gate("nope", 1, matches=trap, settings=settings), "nope", "tracking")
    expect_error(BenchmarkError, lambda: selftrain.gate("clean", 1, matches=trap, settings=settings,
                                                        calibration=synthetic_calibration()), "clean", "calibration")
    expect_error(BenchmarkError, lambda: selftrain.gate("clean", 1, match_ids=["RND009"], matches=trap,
                                                        settings=settings), "RND009")
    expect_error(BenchmarkError, lambda: selftrain.gate("clean", 1, matches={"RND001": matches["RND002"]},
                                                        settings=settings), "selftrain", "RND002", "RND001")
    expect_error(ValueError, lambda: selftrain.gate("clean", 1, matches=matches,
                                                    settings=settings._replace(epsilon=0.5)), "epsilon")
    expect_error(ValueError, lambda: selftrain.gate("clean", 1, matches=matches,
                                                    settings=settings._replace(teacher=dict(TEACHER, hold=0.6))),
                 "teacher", "hold")
    expect_error(ValueError, lambda: selftrain.gate("clean", 1, matches=matches,
                                                    settings=settings._replace(min_prob=0.5)), "min_prob")
    expect_error(BenchmarkError, lambda: selftrain.gate("clean", 1, matches=matches, settings=tuple(settings)),
                 "Settings")
    expect_error(ValueError, lambda: selftrain.gate("clean", 1, matches=matches,
                                                    settings=settings._replace(shape_model="")), "model")


def twisted(match):
    """`match` with its truth flipped, H for A and A for H, on the frames of every other hundred: some, not all."""
    possession = np.array(match.ball_possession)
    flip = ((np.arange(len(possession)) // 100) % 2 == 0) & (possession >= 0)
    possession[flip] = 1 - possession[flip]
    possession.setflags(write=False)
    return dataclasses.replace(match, ball_possession=possession)


def test_selftrain_never_reads_truth_through_inputs():
    root, matches, settings, report = fixture()
    mirror = {"RND001": scene_match("RND001", 0, mirror=True)}
    part = {"RND001": twisted(matches["RND001"])}
    codes, _ = truth.truth_codes(matches["RND001"])
    flipped, _ = truth.truth_codes(mirror["RND001"])
    assert np.array_equal(np.where(codes == H, A, np.where(codes == A, H, codes)), flipped), "the mirror swaps H and A"
    partial, _ = truth.truth_codes(part["RND001"])
    assert 0.2 < float((partial[codes <= 1] != codes[codes <= 1]).mean()) < 0.8, "the other twin flips some truth"
    one = report["per_match"]["RND001"]
    for twin_matches, name in ((mirror, "mirror"), (part, "partial")):
        twin = selftrain.gate("clean", 1, matches=twin_matches, settings=settings)["per_match"]["RND001"]
        # nothing the teacher or the offsets are made of moves: they read the tracking, never the truth
        for variant in ("rung1", "e1", "single", "rung3"):
            assert [c["b"][variant] for c in one["cells"]] == [c["b"][variant] for c in twin["cells"]], (name, variant)
        for key in ("fallback", "rows_home", "rows_away", "pi_hat", "share", "steps"):
            assert [c[key] for c in one["cells"]] == [c[key] for c in twin["cells"]], (name, key)
        for key in ("known_frames", "masked_share", "rows", "rows_home", "rows_away", "home_share", "pi_hat"):
            assert one["teacher"][key] == twin["teacher"][key], (name, key)
        assert one["touch"]["frames"] == twin["touch"]["frames"] > 500, "which frames are kept does not move"
        # what is measured against the truth does move: the precision, the losses, the agreement with the touches
        assert one["teacher"]["precision"] != twin["teacher"]["precision"], name
        for variant in VARIANTS:
            assert one["variants"][variant]["log_loss"] != twin["variants"][variant]["log_loss"], (name, variant)
        if name == "mirror":
            assert one["gate_frames"] == twin["gate_frames"], "which frames are alive does not depend on possession"
            # the base calls the same teams, so its accuracy against the mirror truth is 1 minus its own
            assert abs(one["variants"]["base"]["accuracy"] + twin["variants"]["base"]["accuracy"] - 1.0) <= 1e-12
            # The oracles do not move under a mirror: the class-balanced intercept's score is
            # 1/2 - 1/2 (mean sigma over home + mean sigma over away), which does not care which class is called home.
            for key in ("b_gap", "b_touch"):
                assert [c[key] for c in one["cells"]] == [c[key] for c in twin["cells"]], (name, key)
        else:
            # but they read the truth: a truth that is changed on some stretches only gives other oracles
            assert [c["b_gap"] for c in one["cells"]] != [c["b_gap"] for c in twin["cells"]]
            assert [c["b_touch"] for c in one["cells"]] != [c["b_touch"] for c in twin["cells"]]


def test_selftrain_report_is_aggregates_only():
    root, matches, settings, report = fixture()
    plain = json.loads(json.dumps(benchmark_run.plain(report), allow_nan=False))
    strict_json(json.dumps(benchmark_run.plain(report)))
    table = max(len(report["matches"]), len(report["seeds"]) * 2)
    seen = {"lists": 0, "numbers": 0}

    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str) and len(key) <= 40, (where, key)
                walk(item, "{}.{}".format(where, key))
        elif isinstance(value, list):
            assert len(value) <= max(64, table), "{} is a list of {}: no per-frame list".format(where, len(value))
            assert len(value) <= 64 and not all(isinstance(item, float) for item in value) or len(value) <= 4, where
            seen["lists"] += 1
            for index, item in enumerate(value):
                walk(item, "{}[{}]".format(where, index))
        elif isinstance(value, str):
            assert len(value) <= 400, (where, len(value))
        elif isinstance(value, float):
            seen["numbers"] += 1
            assert math.isfinite(value), where
            assert value == round(value, 9), "{}: floats are written at 9 decimals".format(where)
        else:
            assert value is None or isinstance(value, (bool, int)), (where, type(value))
    walk(plain, "report")
    assert seen["lists"] >= 3 and seen["numbers"] > 100
    lines = selftrain.render_gate(report)
    assert all(len(line) <= 120 for line in lines), [line for line in lines if len(line) > 120]
    text = "\n".join(lines)
    for match_id in IDS:
        assert match_id in text
    for word in ("rung 1", "rung 3", "band", "E/B", "GO"):
        assert word in text, word
    assert not any(str(frame) in text.replace(",", " ").split() for frame in range(1000, 1300)), "a frame was printed"


def test_selftrain_gate_reports_fallbacks_and_goes_on():
    """A block that cannot be fitted is not an exception: b = 0, a reason, and the variant is the base on that block."""
    root, matches, settings, report = fixture()
    one = {"RND001": matches["RND001"]}
    short = {"RND001": scene_match("RND001", 3, rows=(900, 800))}

    def stub_no_root(z, y, *, epsilon):
        raise recalibration.NoRoot("recalibration: stub: no root")

    def stub_no_convergence(*arguments, **more):
        raise selftrain.NoConvergence("selftrain: stub: no convergence")

    real_offset, real_head = recalibration.noise_aware_offset, selftrain.head_fit
    try:
        recalibration.noise_aware_offset = stub_no_root
        rootless = selftrain.gate("clean", 1, matches=one, settings=settings)["per_match"]["RND001"]
        recalibration.noise_aware_offset = real_offset
        selftrain.head_fit = stub_no_convergence
        stuck = selftrain.gate("clean", 1, matches=one, settings=settings)["per_match"]["RND001"]
    finally:
        recalibration.noise_aware_offset, selftrain.head_fit = real_offset, real_head
    # E2 has no root: rung 1, single and rung 3 (whose last step is E2) fall back, E1 does not
    for cell in rootless["cells"]:
        assert cell["fallback"] == {"rung1": "no root", "e1": None, "single": "no root", "rung3": "no root"}, cell
        assert (cell["b"]["rung1"], cell["b"]["single"], cell["b"]["rung3"]) == (0.0, 0.0, 0.0)
        assert cell["b"]["e1"] != 0.0
        assert cell["pi_hat"] is None and cell["steps"] is None and cell["share"] is not None
    for name in ("rung1", "single", "rung3"):
        assert rootless["variants"][name] == rootless["variants"]["base"], name
    assert rootless["variants"]["e1"] != rootless["variants"]["base"]
    assert rootless["fallbacks"]["rung1"] == {"few rows": 0, "no root": 2}
    assert rootless["fallbacks"]["rung3"] == {"few rows": 0, "no root": 2, "no convergence": 0}
    # the head does not converge: rung 3 is the base on that block, and the others are as they were
    assert [c["fallback"]["rung3"] for c in stuck["cells"]] == ["no convergence"] * 2
    assert stuck["variants"]["rung3"] == stuck["variants"]["base"]
    assert stuck["variants"]["rung1"] != stuck["variants"]["base"]
    assert stuck["fallbacks"]["rung3"] == {"few rows": 0, "no root": 0, "no convergence": 2}
    assert [c["b"]["rung1"] for c in stuck["cells"]] == [c["b"]["rung1"]
                                                         for c in report["per_match"]["RND001"]["cells"]]
    # too few rows of a class among the other blocks': the floor, for every variant that is fitted on them
    little = selftrain.gate("clean", 1, matches=short, settings=settings)
    cells = little["per_match"]["RND001"]["cells"]
    assert [c["fallback"][name] for c in cells for name in ("rung1", "e1", "rung3")] == ["few rows"] * 6, cells
    assert all(c["b"][name] == 0.0 for c in cells for name in ("rung1", "e1", "rung3")), cells
    assert all(min(c["rows_home"], c["rows_away"]) < recalibration.MIN_ROWS for c in cells)
    variants = little["per_match"]["RND001"]["variants"]
    assert all(variants[name] == variants["base"] for name in ("rung1", "e1", "rung3"))
    assert little["per_match"]["RND001"]["gate_frames"] > 300, "there is still a gate to measure on"
    assert little["pooled"]["fallbacks"]["rung1"]["few rows"] == little["pooled"]["fallbacks"]["e1"]["few rows"] == 2
    assert little["gate"]["exit_code"] == 1 and little["gate"]["rung1"]["pooled_gain"] == 0.0
    assert not little["gate"]["rung1"]["go"]
    assert any("few rows" in line for line in selftrain.render_gate(little))


def test_selftrain_main_runs_both_commands_end_to_end():
    """`gate` then `bar-ath` through main, with nothing of the tool replaced but where its inputs come from."""
    ids = ("J03WMX", "J03WN1")                    # DFL's own ids, so that the runner knows them; the data is synthetic
    matches = {match_id: scene_match(match_id, seed) for seed, match_id in enumerate(ids)}
    inputs = scene_inputs()
    real_settings, real_inputs = selftrain.read_settings, selftrain.Inputs
    try:
        with tempfile.TemporaryDirectory() as root, loading(lambda match_id: matches[match_id]):
            settings = synthetic_settings(root, ids=ids)
            selftrain.read_settings = lambda: settings
            selftrain.Inputs = lambda data_dir=None: inputs
            flags = ["gate", "--profile", "clean", "--matches", *ids]
            written = []
            for run_number in range(2):
                data = os.path.join(root, "data{}".format(run_number))
                code, text = quietly(selftrain.main, flags + ["--data-dir", data])
                path = os.path.join(data, "runs", "selftrain", "gate-clean.m-J03WMX-J03WN1.json")
                assert os.listdir(os.path.dirname(path)) == [os.path.basename(path)], os.listdir(os.path.dirname(path))
                with open(path, "rb") as handle:
                    written.append(handle.read())
            assert written[0] == written[1], "the same inputs wrote different bytes"
            report = json.loads(written[0])
            assert code == report["gate"]["exit_code"] and report["matches"] == list(ids) and report["seeds"] == [0]
            assert "gate      J03WMX" in text and "verdict" in text and "gate-clean.m-J03WMX-J03WN1.json" in text, text
            assert report["settings"]["epsilon"] == 0.07 and strict_json(written[0].decode("utf-8")) == report
            direct = selftrain.gate("clean", 1, match_ids=list(ids), settings=settings)
            assert benchmark_run.plain(direct) == report, "main writes the report that gate returns, rounded"
            # bar-ath, from the gate report that was just written: its band is read from the file
            data = os.path.join(root, "data0")
            path = os.path.join(data, "runs", "selftrain", "gate-clean.m-J03WMX-J03WN1.json")
            code, text = quietly(selftrain.main, ["bar-ath", "--gate-report", path, "--data-dir", data])
            with open(os.path.join(data, "runs", "selftrain", "bar-ath.json"), encoding="utf-8") as handle:
                barred = json.load(handle)
            assert code == barred["exit_code"], (code, barred["exit_code"])
            assert barred["band"] == report["oracles"]["band"] > 0, barred["band"]
            assert barred["gate_report"] == {"profile": {"name": "clean", "version": 1}, "seeds": [0]}
            assert barred == benchmark_run.plain(selftrain.bar_ath(report, inputs=inputs, settings=settings))
            assert "Smart Tagging" in text and "bar-ath.json" in text and "verdict" in text, text
    finally:
        selftrain.read_settings, selftrain.Inputs = real_settings, real_inputs


# -- BAR-ATH ----------------------------------------------------------------------------------------------------------

def plain_bar_ath(inputs, settings):
    """What bar_ath should say of `inputs`, by loops: per block the cross-fit E2 offset (None where a fallback), the
    teacher's shares and the over-eleven shares; and the Smart Tagging frames with the two agreements."""
    epsilon, min_prob, floor = settings.epsilon, settings.min_prob, recalibration.MIN_ROWS
    run_context = RunContext(inputs=inputs, seed=0)
    path = settings.shape_model
    artifact = shape_possession.load(path)
    z = shape_possession._logit_home(inputs, path, "full", artifact["full"])
    y, rows = recalibration.teacher_labels(run_context, alive_model=settings.alive_model, teacher=settings.teacher)
    alive = recalibration.dead_ball_model.compute(run_context, model=settings.alive_model).alive
    vote = smart_tagging.compute(run_context, c_team=settings.c_team, spread="frame").home
    blocks = [(start, stop) for _, start, stop in inputs.graded]
    present, teams = inputs.atd.players_present, inputs.atd.slot_team
    usable = [bool(rows[f]) and math.isfinite(z[f]) for f in range(len(z))]
    out = []
    for k, (start, stop) in enumerate(blocks):
        pool = [f for j, (a, b) in enumerate(blocks) if j != k for f in range(a, b) if usable[f]]
        homes = sum(int(y[f]) for f in pool)
        offset = 0.0
        if homes >= floor and len(pool) - homes >= floor:
            offset = recalibration.noise_aware_offset(np.array([z[f] for f in pool]),
                                                      np.array([int(y[f]) for f in pool]), epsilon=epsilon).offset
        own = [f for f in range(start, stop) if usable[f]]
        known = [f for f in range(start, stop) if not math.isnan(y[f])]
        home_over = sum(int(present[f][teams == 0].sum() > 11) for f in own)
        away_over = sum(int(present[f][teams == 1].sum() > 11) for f in own)
        out.append({"b": offset, "share": homes / len(pool),
                    "masked": sum(int(not rows[f]) for f in known) / len(known),
                    "home_over": home_over / len(own), "away_over": away_over / len(own), "own": len(own)})
    return out, z, vote, alive, blocks


def test_selftrain_bar_ath_band_and_tag_check():
    root, _, settings, _ = fixture()
    inputs = scene_inputs()
    gate_report = {"profile": {"name": "tracking", "version": 3}, "seeds": [0, 1, 2], "oracles": {"band": 5.0}}
    report = selftrain.bar_ath(gate_report, inputs=inputs, settings=settings)
    plain, z, vote, alive, blocks = plain_bar_ath(inputs, settings)
    assert report["kind"] == "selftrain_bar_ath" and report["band"] == 5.0 and len(report["blocks"]) == 2
    assert report["gate_report"] == {"profile": {"name": "tracking", "version": 3}, "seeds": [0, 1, 2]}
    for got, want, (_, start, stop) in zip(report["blocks"], plain, inputs.graded):
        assert got["half"] in (1, 2) and got["fallback"] is None
        assert got["b"] == want["b"] and abs(got["b"]) > 1e-3, (got, want)
        assert abs(got["share"] - want["share"]) <= 1e-12 and 0.2 < got["share"] < 0.8
        assert 0.0 < got["pi_hat"] < 1.0 and got["rows"] == want["own"] > 500
        assert abs(got["masked_share"] - want["masked"]) <= 1e-12 and got["masked_share"] > 0
        assert got["over_home"] == want["home_over"] and got["over_away"] == want["away_over"]
    assert abs(report["offset_gap"] - abs(plain[0]["b"] - plain[1]["b"])) <= 1e-12
    # the Smart Tagging check: the frames, then the two agreements, by a plain loop
    frames = [f for _, start, stop in inputs.graded for f in range(start, stop)
              if not inputs.atd.ball_present[f] and alive[f] >= 0.5 and not math.isnan(vote[f]) and math.isfinite(z[f])]
    adapted = {f: z[f] + plain[0 if f < blocks[0][1] else 1]["b"] for f in frames}

    def served(logit):
        p = 1.0 / (1.0 + math.exp(-logit))
        return min(max(p, settings.min_prob), 1.0 - settings.min_prob)
    base_share = sum((served(z[f]) >= 0.5) == (vote[f] >= 0.5) for f in frames) / len(frames)
    adapted_share = sum((served(adapted[f]) >= 0.5) == (vote[f] >= 0.5) for f in frames) / len(frames)
    check = report["checks"]["smart_tagging_gaps"]
    assert check["frames"] == len(frames) > 500, (check["frames"], len(frames))
    assert abs(check["base"] - base_share) <= 1e-12 and abs(check["adapted"] - adapted_share) <= 1e-12, check
    assert check["adapted"] != check["base"], "an offset that moves no call proves little: the check has no teeth"
    assert check["margin"] == 0.005 and check["passed"] is bool(round(adapted_share - base_share, 9) >= -0.005)
    assert abs(check["difference"] - (adapted_share - base_share)) <= 1e-12
    assert sum(b["frames"] for b in report["checks"]["smart_tagging_gaps"]["blocks"]) == len(frames)
    assert report["touch"]["frames"] > 500 and 0.0 <= report["touch"]["adapted"] <= 1.0
    assert any(block["over_home"] > 0 for block in report["blocks"]) and any(block["over_away"] > 0
                                                                              for block in report["blocks"])

    # the exit codes: a failed check wins over the band. TAG_MARGIN is read when the check is made, so that it can be
    # set to what always passes (1.0) and what never does (-1.0): adapted >= base - margin
    biggest = max(abs(b["b"]) for b in report["blocks"])
    real = selftrain.TAG_MARGIN
    try:
        outcomes = {}
        for margin in (1.0, -1.0):
            selftrain.TAG_MARGIN = margin
            for band in (biggest + 1e-6, biggest / 2.0, biggest):
                got = selftrain.bar_ath(dict(gate_report, oracles={"band": band}), inputs=inputs, settings=settings)
                outcomes[(margin, band)] = got
                assert got["exit_code"] == selftrain.bar_ath_exit_code(got)
    finally:
        selftrain.TAG_MARGIN = real
    for margin, expected in ((1.0, {biggest + 1e-6: 0, biggest / 2.0: 3, biggest: 0}),
                             (-1.0, {biggest + 1e-6: 1, biggest / 2.0: 1, biggest: 1})):
        for band, code in expected.items():
            got = outcomes[(margin, band)]
            assert got["exit_code"] == code, (margin, band, got["exit_code"], got["failed_checks"], got["outside_band"])
            assert (got["failed_checks"] == []) == (margin == 1.0)
            assert (got["outside_band"] != []) == (band < biggest), (band, got["outside_band"])
    assert outcomes[(1.0, biggest / 2.0)]["outside_band"] == [b["half"] for b in report["blocks"]
                                                              if abs(b["b"]) > biggest / 2.0]
    assert outcomes[(-1.0, biggest / 2.0)]["failed_checks"] == ["smart_tagging_gaps"]
    # the verdict function on its own: the check first, then the band
    assert [selftrain.bar_ath_verdict(failed, outside)[0] for failed, outside in
            (([], []), ([], [2]), (["x"], []), (["x"], [1, 2]))] == [0, 3, 1, 1]
    assert "STOP" in selftrain.bar_ath_verdict([], [2])[1] and "whatever the band says" in selftrain.bar_ath_verdict(
        ["x"], [2])[1]
    # the band is read from the gate report, a number, or an error that says where
    for bad in ({}, {"oracles": {}}, {"oracles": {"band": None}}, {"oracles": {"band": -0.1}},
                {"oracles": {"band": float("nan")}}, {"oracles": {"band": True}}):
        expect_error(BenchmarkError, lambda: selftrain.bar_ath(bad, inputs=inputs, settings=settings), "selftrain",
                     "band", "gate")
    # lines of figures, no frame
    lines = selftrain.render_bar_ath(report)
    assert all(len(line) <= 140 for line in lines) and any("Smart Tagging" in line for line in lines)
    assert any("verdict" in line for line in lines) and len(lines) <= 14, len(lines)
    # a match too short for the floor falls back on both blocks, says so, and leaves P07 as it is: offset 0, the same
    # agreement, no pi_hat, and nothing outside the band
    short = selftrain.bar_ath(gate_report, inputs=scene_inputs(7, rows=(900, 800)), settings=settings)
    assert [block["fallback"] for block in short["blocks"]] == ["few rows", "few rows"], short["blocks"]
    assert [block["b"] for block in short["blocks"]] == [0.0, 0.0] and short["blocks"][0]["pi_hat"] is None
    assert short["checks"]["smart_tagging_gaps"]["difference"] == 0.0 and short["exit_code"] == 0
    assert short["blocks"][0]["rows_home"] < 500 or short["blocks"][0]["rows_away"] < 500
    assert any("few rows" in line for line in selftrain.render_bar_ath(short))


def test_selftrain_tag_check_at_the_margin():
    n = 1000
    vote = np.where(np.arange(n) % 2 == 0, 0.76, 0.24)                  # home on the even frames, away on the odd
    agrees = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)                 # a logit that names the vote's team

    def logits(right):
        """A logit array that names the vote's team on `right` of the frames and the other team on the rest."""
        out = agrees * 0.5
        out[right:] = -out[right:]
        return out
    frames = np.ones(n, dtype=bool)
    base = logits(800)
    for adapted_right, passes in ((804, True), (800, True), (796, True), (795, True), (794, False), (790, False)):
        got = selftrain.tag_check(vote, frames, base, logits(adapted_right), 0.02)
        assert got["frames"] == n and got["base"] == 0.8 and got["adapted"] == adapted_right / n, got
        assert got["passed"] is passes, (adapted_right, got)
        assert abs(got["difference"] - (adapted_right - 800) / n) <= 1e-12 and got["margin"] == 0.005
    # at base - 0.004 it passes and at base - 0.006 it fails; the edge itself, -0.005, passes: adapted >= base - margin
    assert selftrain.tag_check(vote, frames, base, logits(796), 0.02)["passed"]
    assert not selftrain.tag_check(vote, frames, base, logits(794), 0.02)["passed"]
    assert selftrain.tag_check(vote, frames, base, logits(795), 0.02)["passed"]
    assert selftrain.tag_check(vote, frames, base, logits(794), 0.02, margin=0.006)["passed"]
    # only the masked frames count; a frame the mask leaves out cannot move either share
    half = frames.copy()
    half[500:] = False
    got = selftrain.tag_check(vote, half, base, logits(400), 0.02)
    assert got["frames"] == 500 and got["base"] == 1.0 and got["adapted"] == 0.8, got
    # the clip does not move a call: sigma(z) >= 0.5 where z >= 0, with and without the floor
    assert selftrain.tag_check(vote, frames, base, base * 30.0, 0.4)["adapted"] == 0.8
    # no frame, no check: an error, since the check would measure nothing
    expect_error(ValueError, lambda: selftrain.tag_check(vote, np.zeros(n, dtype=bool), base, base, 0.02), "selftrain",
                 "frame")
    expect_error(ValueError, lambda: selftrain.tag_check(vote[:-1], frames, base, base, 0.02), "selftrain", "one value")


# -- the command line -------------------------------------------------------------------------------------------------

def test_selftrain_command_line_exit_codes():
    root, matches, settings, report = fixture()
    calls = []

    def fake_gate(code):
        def gate(profile, seeds, match_ids=None, data_dir=None, log=None, **more):
            calls.append((profile, seeds, match_ids, data_dir))
            if isinstance(code, Exception):
                raise code
            made = copy.deepcopy(report)
            made["gate"]["exit_code"] = code
            return made
        return gate

    real_gate, real_bar = selftrain.gate, selftrain.bar_ath
    try:
        with tempfile.TemporaryDirectory() as data:
            for code in (0, 4, 1):
                selftrain.gate = fake_gate(code)
                got, text = quietly(selftrain.main, ["gate", "--data-dir", data])
                assert got == code, (code, got, text)
                path = os.path.join(data, "runs", "selftrain", "gate-tracking.json")
                with open(path, encoding="utf-8") as handle:
                    written = handle.read()
                assert written.endswith("\n") and "\r" not in written
                assert "report" in text and "gate-tracking.json" in text
                assert json.loads(written)["gate"]["exit_code"] == code and calls[-1][:2] == ("tracking", 3)
                strict_json(written)
            # a quick look does not replace the real report: the benchmark's own suffixes for matches and seeds
            selftrain.gate = fake_gate(0)
            quietly(selftrain.main, ["gate", "--profile", "context", "--seeds", "1", "--matches", "J03WMX",
                                     "--data-dir", data])
            assert sorted(os.listdir(os.path.join(data, "runs", "selftrain"))) == [
                "gate-context.m-J03WMX.s1.json", "gate-tracking.json"]
            before = sorted(os.listdir(os.path.join(data, "runs", "selftrain")))
            got, text = quietly(selftrain.main, ["gate", "--no-write", "--data-dir", data])
            assert got == 0 and "not written" in text
            assert sorted(os.listdir(os.path.join(data, "runs", "selftrain"))) == before

            # an error is 2, with a sentence: a crash must never look like a verdict (python's own exit status for an
            # uncaught exception is 1, which here would read as "both NO-GO" and as "a check failed")
            for failure in (BenchmarkError("selftrain: the fold is missing"), OSError("no such file"),
                            RuntimeError("a bug")):
                selftrain.gate = fake_gate(failure)
                shown = io.StringIO()
                with contextlib.redirect_stdout(shown), contextlib.redirect_stderr(shown):
                    got = selftrain.main(["gate", "--no-write"])
                assert got == 2 and "FAILED" in shown.getvalue(), (failure, got, shown.getvalue())
            assert "Traceback" in shown.getvalue() and "a bug" in shown.getvalue(), "a bug keeps its traceback"

            # bar-ath: the gate's report is read from its path; 0, 3 and 1 are verdicts and an unreadable report is 2
            gate_path = os.path.join(data, "gate.json")
            with open(gate_path, "w", encoding="utf-8") as handle:
                json.dump({"oracles": {"band": 0.5}}, handle)
            got_args = []

            def fake_bar(code):
                def bar(gate_report, data_dir=None, inputs=None, settings=None, **more):
                    got_args.append((gate_report, data_dir))
                    return {"kind": "selftrain_bar_ath", "exit_code": code, "blocks": [], "verdict": "x"}
                return bar
            for code in (0, 3, 1):
                selftrain.bar_ath = fake_bar(code)
                selftrain.render_bar_ath, real_render = (lambda result: ["bar-ath   stub"]), selftrain.render_bar_ath
                try:
                    got, text = quietly(selftrain.main, ["bar-ath", "--gate-report", gate_path, "--data-dir", data])
                finally:
                    selftrain.render_bar_ath = real_render
                assert got == code and got_args[-1] == ({"oracles": {"band": 0.5}}, data), (code, got, text)
                assert os.path.exists(os.path.join(data, "runs", "selftrain", "bar-ath.json"))
            selftrain.bar_ath = real_bar
            for path in (os.path.join(data, "absent.json"), write_text(data, "broken.json", "{not json")):
                shown = io.StringIO()
                with contextlib.redirect_stdout(shown):
                    got = selftrain.main(["bar-ath", "--gate-report", path, "--no-write"])
                assert got == 2 and "FAILED" in shown.getvalue() and os.path.basename(path) in shown.getvalue()
    finally:
        selftrain.gate, selftrain.bar_ath = real_gate, real_bar
    assert selftrain.report_path("D", "gate", "tracking") == os.path.join(
        "D", "runs", "selftrain", "gate-tracking.json")
    assert selftrain.report_path("D", "bar-ath") == os.path.join("D", "runs", "selftrain", "bar-ath.json")
    assert selftrain.report_path("D", "gate", "clean", ["B", "A"], 1) == os.path.join(
        "D", "runs", "selftrain", "gate-clean.m-A-B.json"), "the clean profile has no seeds to count"
