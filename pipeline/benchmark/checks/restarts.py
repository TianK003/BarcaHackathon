"""Checks of restarts (pipeline/benchmark/restarts.py): the anatomy of the stoppages, on the benchmark and on BAR-ATH.

restarts says how the dead-ball runs of the truth end (which team restarts, how often it is not the team that had the
ball before, how often the pipeline names the restarting team in the 2 s after) and, label-free, what the predicted
stoppages of BAR-ATH look like to the ball. A mistake in it would size P17's restart-team channel wrongly, so what is
held against it here is the restarts found on a hand-built truth (two stoppages, one cut off by an X run), the three
outcomes worked out on hand-built predictions (a tie is wrong, the window never reaches into the next block), the ball
shares on a hand-built scene, the pooling over matches and seeds against the same figures taken one match and seed at a
time, and that the pipeline is never handed the truth: the inputs it is called on go through the deep walk of
common.leaks.

The matches are the synthetic ones of checks/common.py (random flags and tracking, figures meaningless), under the
gaps profile and the synthetic Calibration.
"""

import math
import os
import tempfile

import numpy as np

from pipeline.benchmark import restarts, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import A, D, H, X, leaks, synthetic_calibration, synthetic_matches, yardstick_path
from pipeline.config import load_config
from pipeline.selftest import quiet

# Params
SEEDS = 2


def anatomy_of_synthetic(**more):
    return restarts.anatomy(yardstick_path(50), profile="gaps", seeds=SEEDS, matches=synthetic_matches(),
                            calibration=synthetic_calibration(), log=quiet, **more)


def test_restarts_true_restarts_finds_each_stoppage():
    """Each D run of a block followed by H or A is one restart; one followed by X, or by the block's end, is not."""
    codes = np.array([D] * 3 + [H] * 7 + [D] * 5 + [A] * 5 + [D] * 5 + [X] * 5, dtype=np.int8)   # block [0, 30)
    found = restarts.true_restarts(codes, [(1, 0, 30)])
    assert found["start"].tolist() == [0, 10] and found["restart"].tolist() == [3, 15]
    assert found["before"].tolist() == [-1, H] and found["after"].tolist() == [H, A]
    assert all(found[key].dtype == np.int64 for key in ("start", "restart", "before", "after"))
    # a stoppage that runs to the end of its block has no restart, and the next block's first frame is no "after"
    codes = np.array([H] * 4 + [D] * 4 + [A] * 4 + [D] * 4, dtype=np.int8)
    found = restarts.true_restarts(codes, [(1, 0, 12), (2, 12, 16)])
    assert found["start"].tolist() == [4] and found["before"].tolist() == [H] and found["after"].tolist() == [A]
    # the frame before a stoppage that opens a block is outside it: before is -1 even if the code there is H
    codes = np.array([H] * 4 + [D] * 4 + [A] * 4, dtype=np.int8)
    found = restarts.true_restarts(codes, [(1, 4, 12)])
    assert found["before"].tolist() == [-1] and found["after"].tolist() == [A]


def test_restarts_outcomes_three_ways():
    """Right is more of the restarting team than of the other in the window; a tie is wrong; no H/A is still D."""
    predicted = np.full(80, X, dtype=np.int8)
    predicted[5:9] = (H, H, A, H)            # right
    predicted[15:19] = (A, A, H, D)          # wrong: more of the other team
    predicted[25:29] = D                     # still D
    predicted[38:40] = D
    predicted[40:44] = H                     # the next block's: never counted for a restart of the first
    predicted[50:54] = (H, H, A, A)          # a tie
    found = {"start": np.array([3, 13, 23, 36, 48], dtype=np.int64),
             "restart": np.array([5, 15, 25, 38, 50], dtype=np.int64),
             "before": np.array([-1, A, H, -1, A], dtype=np.int64),
             "after": np.array([H, H, A, H, H], dtype=np.int64)}
    out = restarts.outcomes(predicted, found, [(1, 0, 40), (2, 40, 80)], window=4)
    assert out.tolist() == [0, 1, 2, 2, 1], out.tolist()
    assert out.dtype == np.int8
    assert restarts.OUTCOME_WINDOW == 50 and restarts.BALL_WINDOW == 50


def test_restarts_stoppage_ball_shares():
    """Two predicted stoppages; the ball missing on 3 of the last 4 frames before the first only."""
    codes = np.array([H] * 8 + [D] * 4 + [A] * 8 + [D] * 4 + [H] * 4 + [D] * 4, dtype=np.int8)   # block [0, 32)
    visible = np.ones(32, dtype=bool)
    visible[[8, 9, 10]] = False              # inside the first stoppage: not in any window
    visible[[4, 5, 6]] = False               # 3 of the 4 frames before the first stoppage
    visible[[12, 13]] = False                # 2 of 4 after the first: not more than half
    visible[[24, 25, 26]] = False            # 3 of the 4 frames after the second
    shares = restarts.stoppage_ball_shares(codes, visible, [(1, 0, 32)], window=4)
    assert shares["stoppages"] == 2 and shares["per_half"] == {1: 2}, shares
    assert shares["no_ball_before"] == 0.5 and shares["no_ball_after"] == 0.5, shares
    # a stoppage that opens its block has no frame before it: it is left out of that share
    shares = restarts.stoppage_ball_shares(codes[8:], visible[8:], [(1, 0, 24)], window=4)
    assert shares["stoppages"] == 2 and shares["no_ball_before"] == 0.0, shares
    # none
    shares = restarts.stoppage_ball_shares(np.full(10, H, dtype=np.int8), np.ones(10, dtype=bool), [(1, 0, 10)])
    assert shares["stoppages"] == 0 and shares["no_ball_before"] is None and shares["per_half"] == {}, shares


def test_restarts_anatomy_pools_over_matches_and_seeds():
    """Pooled restarts are the sum over matches and seeds of the truth's restarts; the shares are those of the sums."""
    matches = synthetic_matches()
    result = anatomy_of_synthetic()
    assert result["matches"] == sorted(matches) and result["seeds"] == [0, 1] and result["truth"] == "raw"
    total = flips = known = 0
    outcome_totals = np.zeros(3, dtype=int)
    calibration = synthetic_calibration()
    config = load_config(yardstick_path(50))
    for match_id, match in matches.items():
        codes, _ = truth.truth_codes(match)
        found = restarts.true_restarts(codes, match.graded)
        assert len(found["start"]) > 3, "a match with no restart proves nothing"
        row = result["per_match"][match_id]
        assert row["restarts"] == SEEDS * len(found["start"]), row
        assert row["median_stoppage_s"] == float(np.median((found["restart"] - found["start"]) / match.fps)), row
        counts = np.zeros(3, dtype=int)
        for seed in range(SEEDS):
            atd = benchmark_run.make_atd(match, "gaps", seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
            predicted, _ = benchmark_run.predict(config, inputs)
            counts += np.bincount(restarts.outcomes(predicted, found, match.graded), minlength=3)
        assert counts.sum() == row["restarts"]
        for name, count in zip(("right", "wrong", "still_d"), counts):
            assert row[name] == count / row["restarts"], (match_id, name, row)
        known_here = found["before"] != -1
        known += SEEDS * int(known_here.sum())
        flips += SEEDS * int((found["after"][known_here] != found["before"][known_here]).sum())
        total += row["restarts"]
        outcome_totals += counts
    pooled = result["pooled"]
    assert pooled["restarts"] == total and 0 < flips < known
    assert pooled["flip_share"] == flips / known, pooled
    for name, count in zip(("right", "wrong", "still_d"), outcome_totals):
        assert pooled[name] == count / total, (name, pooled)
    assert pooled["flip"]["restarts"] == flips and pooled["flip"]["restarts"] + pooled["no_flip"]["restarts"] == known
    for part in ("flip", "no_flip"):
        shares = [pooled[part][name] for name in ("right", "wrong", "still_d")]
        assert abs(sum(shares) - 1.0) < 1e-12, (part, shares)


def test_restarts_coarse_truth_keeps_the_stoppages():
    """The coarse truth never moves a stoppage, so the restarts are the raw truth's; only the teams may differ."""
    raw, coarse = anatomy_of_synthetic(), anatomy_of_synthetic(truth="coarse")
    assert coarse["truth"] == "coarse"
    assert raw["pooled"]["restarts"] == coarse["pooled"]["restarts"]


def test_restarts_never_pass_truth():
    """The inputs the pipeline is called on hold neither the match nor its truth, however deep leaks() looks."""
    matches = synthetic_matches()
    seen = []
    original = restarts.run.predict

    def spy(config, inputs):
        seen.append(inputs)
        return original(config, inputs)

    restarts.run.predict = spy
    try:
        anatomy_of_synthetic()
    finally:
        restarts.run.predict = original
    assert len(seen) == len(matches) * SEEDS, "one call per match and seed"
    for inputs, match in zip(seen, [m for m in matches.values() for _ in range(SEEDS)]):
        assert leaks(inputs, match) == [], leaks(inputs, match)


# -- stats and events (P17): the restart vote's confidence per type, measured on the benchmark ------------------------

def fake_calls(run, *, model, tail, use_ball=False, setup=50, min_stoppage=25, min_prob=0.001):
    """A stand-in for restart_team.calls: a call every 37 frames, types and teams cycling, free kicks only with the ball.

    It reads only the inputs' length, so the stats' own counting is what the tests hold against a plain loop; the type
    and team do not depend on `tail`, and `end` does, as in the real source.
    """
    n = run.inputs.atd.n_frames
    restart = np.arange(20, n - 5, 37, dtype=np.int64)
    kinds = len(restarts.restart_team.TYPES) if use_ball else len(restarts.restart_team.TYPES) - 1
    kind = (np.arange(len(restart)) % kinds).astype(np.int8)
    team = ((np.arange(len(restart)) // 2) % 2).astype(np.int8)
    following = np.append(restart[1:], n)
    return {"restart": restart, "end": np.minimum(restart + tail, following), "type": kind, "team": team}


class FakeCalls:
    """restarts.restart_team.calls replaced by fake_calls, recording what it was called with."""

    def __init__(self):
        self.runs, self.kwargs = [], []

    def __enter__(self):
        self.real = restarts.restart_team.calls

        def spy(run, **kwargs):
            self.runs.append(run)
            self.kwargs.append(kwargs)
            return fake_calls(run, **kwargs)
        restarts.restart_team.calls = spy
        return self

    def __exit__(self, *exc):
        restarts.restart_team.calls = self.real


def stats_of_synthetic(settings, **more):
    with FakeCalls() as fake:
        result = restarts.measure_restarts(settings, model="unused.json", profile="gaps", seeds=SEEDS,
                                           matches=synthetic_matches(), calibration=synthetic_calibration(),
                                           log=quiet, **more)
    return result, fake


def test_restarts_stats_grid():
    """Eight settings, tail within use_ball: the labels t025_b0 ... t150_b1, one entry per type."""
    assert [restarts.label(setting) for setting in restarts.GRID] == [
        "t{:03d}_b{}".format(tail, ball) for ball in (0, 1) for tail in (25, 50, 100, 150)]
    assert restarts.GRID[0] == {"tail": 25, "use_ball": False} and restarts.GRID[-1] == {"tail": 150, "use_ball": True}
    assert (restarts.MIN_AGREEMENT, restarts.MIN_STOPPAGES_PER_SEED, restarts.COVERAGE_WINDOW) == (0.60, 30, 50)
    result, _ = stats_of_synthetic(restarts.GRID)
    assert list(result["settings"]) == [restarts.label(setting) for setting in restarts.GRID]
    for name, entry in result["settings"].items():
        assert list(entry["types"]) == list(restarts.restart_team.TYPES), name
        assert entry["setting"] in [dict(setting) for setting in restarts.GRID]
    # the tail is the vote's length, so a longer tail votes on at least as many frames
    short, long = result["settings"]["t025_b0"]["types"], result["settings"]["t150_b0"]["types"]
    assert all(long[kind]["frames"] >= short[kind]["frames"] for kind in short)
    assert any(long[kind]["frames"] > short[kind]["frames"] for kind in short)


def test_restarts_stats_matches_a_loop():
    """Frames, agreement, stoppages per seed and coverage equal a plain-loop recount over matches and seeds."""
    settings = [{"tail": 50, "use_ball": False}, {"tail": 100, "use_ball": True}]
    result, _ = stats_of_synthetic(settings)
    calibration = synthetic_calibration()
    types = restarts.restart_team.TYPES
    for setting in settings:
        frames, agree, stoppages = [0] * len(types), [0] * len(types), [0] * len(types)
        covered, owed = [0] * len(types), 0
        for match in synthetic_matches().values():
            codes, _ = truth.truth_codes(match)
            found = restarts.true_restarts(codes, match.graded)
            for seed in range(SEEDS):
                atd = benchmark_run.make_atd(match, "gaps", seed, calibration)
                inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
                made = fake_calls(restarts.RunContext(inputs=inputs, seed=0), model="x", **setting)
                for number in range(len(made["restart"])):
                    kind, team = int(made["type"][number]), int(made["team"][number])
                    stoppages[kind] += 1
                    for frame in range(int(made["restart"][number]), int(made["end"][number])):
                        if int(codes[frame]) in (H, A):
                            frames[kind] += 1
                            agree[kind] += int(int(codes[frame]) == (H if team == 0 else A))
                for restart in found["restart"].tolist():
                    owed += 1
                    for kind in range(len(types)):
                        covered[kind] += int(any(
                            int(made["type"][number]) == kind and abs(int(made["restart"][number]) - restart) <= 50
                            for number in range(len(made["restart"]))))
        entry = result["settings"][restarts.label(setting)]
        assert owed > 0
        for kind, name in enumerate(types):
            row = entry["types"][name]
            assert row["frames"] == frames[kind] and row["agree"] == agree[kind], (setting, name, row)
            assert row["stoppages_per_seed"] == stoppages[kind] / SEEDS, (setting, name, row)
            assert row["coverage"] == covered[kind] / owed, (setting, name, row)
            assert row["agreement"] == (agree[kind] / frames[kind] if frames[kind] else None), (setting, name, row)
        assert 0.0 <= entry["coverage_all"] <= 1.0
        assert (entry["types"]["free_kick"]["stoppages_per_seed"] == 0) == (not setting["use_ball"])


def test_restarts_stats_drop_rule():
    """Agreement under 0.60, or under 30 stoppages per seed, drops the type; otherwise the floored agreement."""
    assert restarts.confidence_of(0.59, 100.0) is None
    assert restarts.confidence_of(0.60, 100.0) == 0.60
    assert restarts.confidence_of(0.60, 30.0) == 0.60
    assert restarts.confidence_of(0.97, 29.0) is None
    assert restarts.confidence_of(0.9999, 40.0) == 0.99
    assert restarts.confidence_of(None, 40.0) is None


def test_restarts_stats_never_passes_truth():
    """The source is called on inputs that hold neither the match nor its truth, however deep leaks() looks."""
    matches = synthetic_matches()
    _, fake = stats_of_synthetic([{"tail": 50, "use_ball": False}])
    assert len(fake.runs) == len(matches) * SEEDS, "one call per match, seed and setting"
    for run_context, match in zip(fake.runs, [m for m in matches.values() for _ in range(SEEDS)]):
        assert leaks(run_context.inputs, match) == [], leaks(run_context.inputs, match)
    assert all(set(kwargs) <= {"model", "tail", "use_ball"} for kwargs in fake.kwargs), fake.kwargs


def test_restarts_events_recall_and_confusion():
    """Recall per DFL kind and the call-type by event-kind table, worked out on a hand-built scene."""
    calls = {"restart": np.array([100, 300, 500], dtype=np.int64), "end": np.array([150, 350, 550], dtype=np.int64),
             "type": np.array([1, 3, 0], dtype=np.int8), "team": np.zeros(3, dtype=np.int8)}
    events = {"row": np.array([95, 360, 460, 505, 900], dtype=np.int64),
              "kind": np.array(["CornerKick", "ThrowIn", "GoalKick", "KickOff", "FreeKick"]),
              "team": np.zeros(5, dtype=np.int8), "outside": 0}
    table = restarts.event_table(calls, events, window=50)
    # 95 -> call 100 (corner); 360 is 60 from 300: no call; 460 -> 500 (kick_off, 40); 505 -> 500 (kick_off); 900: none
    assert table["events"] == {"KickOff": 1, "ThrowIn": 1, "CornerKick": 1, "GoalKick": 1, "FreeKick": 1, "Penalty": 0}
    assert table["matched"] == {"KickOff": 1, "ThrowIn": 0, "CornerKick": 1, "GoalKick": 1, "FreeKick": 0, "Penalty": 0}
    assert table["confusion"]["corner"]["CornerKick"] == 1
    assert table["confusion"]["kick_off"]["GoalKick"] == 1 and table["confusion"]["kick_off"]["KickOff"] == 1
    assert sum(sum(row.values()) for row in table["confusion"].values()) == 3


def test_restarts_configs_stoppage_end_and_restart_states():
    """R014b's configs: the span block without a tail, the t025 confidences, and the decoder's two restart keys."""
    main_path = os.path.join(restarts.ROOT, "pipeline", "configs", "main.toml")
    with open(main_path, newline="") as handle:
        main_text = handle.read()
    main = load_config(main_path)
    eol = "\r\n" if "\r\n" in main_text else "\n"
    types = ("kick_off", "corner", "goal_kick", "throw_in", "free_kick")
    t25 = {0: {"kick_off": 0.91, "corner": 0.7, "throw_in": 0.8}, 1: {"corner": 0.66, "free_kick": 0.7}}

    def setting(tail, use_ball, confidences):
        return {"setting": {"tail": tail, "use_ball": use_ball},
                "types": {name: {"confidence": confidences.get(name)} for name in types}}

    stats = {"settings": {"t025_b0": setting(25, False, t25[0]), "t025_b1": setting(25, True, t25[1]),
                          "t050_b0": setting(50, False, {"corner": 0.5}), "t050_b1": setting(50, True, {"corner": 0.5})}}
    with tempfile.TemporaryDirectory() as root:
        written = restarts.write_configs(stats, main_path, root, log=quiet, span="stoppage_end",
                                         switch_restart=[4, float("inf")], label_from=25)
        names = sorted(os.path.basename(path) for path in written)
        assert names == ["r014b_s04_b0.toml", "r014b_s04_b1.toml", "r014b_sin_b0.toml", "r014b_sin_b1.toml"], names
        for path in written:
            name = os.path.basename(path)
            use_ball = name.endswith("_b1.toml")
            switch = float("inf") if "_sin_" in name else 4.0
            config = load_config(path)
            assert config.evidence[:-1] == main.evidence and len(config.evidence) == len(main.evidence) + 1
            added = config.evidence[-1]
            expected = {"model": "pipeline/models/dead_ball_hgb.json", "span": "stoppage_end", "use_ball": use_ball}
            expected.update(t25[int(use_ball)])
            assert added.kind == "restart_team" and added.weight == 1.0 and added.params == expected, added
            assert "tail" not in added.params
            assert config.decoder.params == dict(main.decoder.params, restart_source="restart_team",
                                                 switch_restart=switch), config.decoder
            assert math.isinf(config.decoder.params["switch_restart"]) == (switch == float("inf"))
            assert config.refine == main.refine and config.seed == main.seed
            assert "stoppage_end" in config.description and "switch_restart" in config.description
            with open(path, newline="") as handle:
                text = handle.read()
            start = text.index("# R014b")
            stop = text.index(eol + eol, start) + 2 * len(eol)
            restored = text[:start] + text[stop:]
            restored = restored.replace(config.description, main.description)
            keys = ['restart_source = "restart_team"' + eol,
                    "switch_restart = {}".format("inf" if switch == float("inf") else "4.0") + eol]
            for key in keys:
                assert restored.count(key) == 1
                restored = restored.replace(key, "")
            assert restored == main_text
        # without switch_restart: the evidence only, the decoder untouched
        plain = restarts.write_configs(stats, main_path, os.path.join(root, "plain"), log=quiet, span="stoppage_end",
                                       label_from=25)
        assert sorted(os.path.basename(path) for path in plain) == ["r014b_b0.toml", "r014b_b1.toml"]
        assert load_config(plain[0]).decoder == main.decoder
        for kwargs in ({"span": "stoppage_end"}, {"span": "after", "switch_restart": [4]}, {"span": "middle"}):
            try:
                restarts.write_configs(stats, main_path, root, log=quiet, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("{} must be refused".format(kwargs))


def test_restarts_configs_insert_one_block():
    """R014's configs are main.toml's text plus one restart_team block and an amended description, and nothing else."""
    main_path = os.path.join(restarts.ROOT, "pipeline", "configs", "main.toml")
    with open(main_path, newline="") as handle:
        main_text = handle.read()
    main = load_config(main_path)
    eol = "\r\n" if "\r\n" in main_text else "\n"
    types = ("kick_off", "corner", "goal_kick", "throw_in", "free_kick")

    def setting(tail, use_ball, confidences):
        return {"setting": {"tail": tail, "use_ball": use_ball},
                "types": {name: {"confidence": confidences.get(name)} for name in types}}

    some = {"kick_off": 0.9, "corner": 0.75, "goal_kick": None, "throw_in": 0.8, "free_kick": 0.7}
    stats = {"settings": {"t050_b0": setting(50, False, some), "t025_b1": setting(25, True, {"corner": 0.66}),
                          "t100_b1": setting(100, True, {})}}
    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "made", "here")
        written = restarts.write_configs(stats, main_path, folder, log=quiet)
        assert sorted(os.path.basename(path) for path in written) == ["r014_t025_b1.toml", "r014_t050_b0.toml"], written
        for path, tail, use_ball, confidences in ((os.path.join(folder, "r014_t050_b0.toml"), 50, False, some),
                                                  (os.path.join(folder, "r014_t025_b1.toml"), 25, True,
                                                   {"corner": 0.66})):
            assert path in written
            config = load_config(path)
            assert config.evidence[:-1] == main.evidence and len(config.evidence) == len(main.evidence) + 1
            added = config.evidence[-1]
            expected = {"model": "pipeline/models/dead_ball_hgb.json", "tail": tail, "use_ball": use_ball}
            expected.update({name: value for name, value in confidences.items() if value is not None})
            assert added.kind == "restart_team" and added.weight == 1.0 and added.params == expected, added
            assert config.decoder == main.decoder and config.refine == main.refine and config.seed == main.seed
            assert config.description.startswith(main.description[:-1]) and "restart_team" in config.description
            assert "tail {}".format(tail) in config.description
            with open(path, newline="") as handle:
                text = handle.read()
            start = text.index("# R014")
            stop = text.index(eol + eol, start) + 2 * len(eol)   # the block ends at its blank line
            assert text.count("[[evidence]]") == main_text.count("[[evidence]]") + 1
            restored = text[:start] + text[stop:]
            restored = restored.replace(config.description, main.description)
            assert restored == main_text
        for bad in ("[decoder]", "description = "):
            broken = os.path.join(root, "broken.toml")
            with open(broken, "w", newline="") as handle:
                handle.write(main_text.replace(bad, "#"))
            try:
                restarts.write_configs(stats, broken, folder, log=quiet)
            except ValueError as error:
                assert "broken.toml" in str(error)
            else:
                raise AssertionError("a main.toml without {!r} must be refused".format(bad))
