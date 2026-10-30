"""Checks of tag_stats (pipeline/benchmark/tag_stats.py): the Smart Tagging votes against our own pseudo-labels.

The agreement decided whether the votes entered the pipeline (R017), so every figure is worked out here by hand on a
small scene: the share, the two coverages, the tag-level bootstrap (every draw of one right
tag and one wrong tag is 0, 0.5 or 1, which a frame-level bootstrap would not give), and the printout, which holds
codes, counts and shares and no frame (CLAUDE.md, hard rules). No check reads data.
"""

import numpy as np

from pipeline.benchmark import tag_stats
from pipeline.benchmark.train_dead_ball import towards_half
from pipeline.selftest import expect_error

# Params
N = 3000
BLOCKS = [(1, 1000, 1600), (2, 2000, 2600)]     # four-digit edges, so that a printed frame would be found


def scene(ball_missing=()):
    graded = np.zeros(N, dtype=bool)
    for _, start, stop in BLOCKS:
        graded[start:stop] = True
    missing = np.zeros(N, dtype=bool)
    for start, stop in ball_missing:
        missing[start:stop] = True
    return graded, missing


def team_scene():
    """Two team tags: A on 1000-1100 (home) over a home episode that turns away at 1050; B on 1200-1300 naming away
    over an away episode. A home episode also covers 1400-1450, where no tag votes."""
    team = np.full(N, -1, dtype=np.int8)
    team[1000:1100] = 1
    team[1200:1300] = 0
    touch = np.full(N, np.nan)
    touch[1000:1050] = 0.93
    touch[1050:1100] = 0.07
    touch[1200:1300] = 0.07
    touch[1400:1450] = 0.93
    tags = [("team", 1, [(1000, 1100)]), ("team", 0, [(1200, 1300)])]
    return team, touch, tags


def test_tag_stats_team_agreement_exact():
    team, touch, tags = team_scene()
    graded, missing = scene(ball_missing=[(1000, 1500)])
    out = tag_stats.team_agreement(team, touch, tags, graded=graded, ball_missing=missing, draws=50)
    # kept frames with a vote: 100 + 100; agreed: 50 (A's first half) + 100 (B)
    assert out["n"] == 200, "n is {}".format(out["n"])
    assert out["share"] == 150 / 200, "share is {}".format(out["share"])
    assert out["coverage"] == 200 / 1200, "coverage is {}".format(out["coverage"])
    # ball-missing graded frames: 1000-1500 is 500, of which 200 have a vote
    assert out["coverage_gaps"] == 200 / 500, "coverage_gaps is {}".format(out["coverage_gaps"])
    assert out["low"] <= out["share"] <= out["high"], "the interval holds the point: {}".format(out)


def test_tag_stats_bootstrap_resamples_whole_tags():
    # one tag all right, one all wrong, equal sizes: a draw is two right, two wrong or one of each, never a mixture
    team = np.full(N, -1, dtype=np.int8)
    team[1000:1100] = 1
    team[1200:1300] = 1
    touch = np.full(N, np.nan)
    touch[1000:1100] = 0.93
    touch[1200:1300] = 0.07
    tags = [("team", 1, [(1000, 1100)]), ("team", 1, [(1200, 1300)])]
    graded, missing = scene()
    draws = tag_stats.team_draws(team, touch, tags, graded=graded, draws=300, seed=(17, 0))
    assert set(np.round(draws, 12)) == {0.0, 0.5, 1.0}, "draws are {}".format(sorted(set(draws)))
    out = tag_stats.team_agreement(team, touch, tags, graded=graded, ball_missing=missing, draws=300)
    assert (out["low"], out["high"]) == (0.0, 1.0), "the interval of two tags, one right one wrong: {}".format(out)


def test_tag_stats_bootstrap_keeps_a_tags_pieces_together():
    # one tag across both blocks is one tag (200 right frames), beside a wrong one (100 frames): a draw takes both of
    # its pieces or neither, so the shares are 0, 2/3 or 1 and never what one piece alone would give
    team = np.full(N, -1, dtype=np.int8)
    team[1500:1600] = 1
    team[2000:2100] = 1
    team[1200:1300] = 1
    touch = np.full(N, np.nan)
    touch[1500:1600] = 0.93
    touch[2000:2100] = 0.93
    touch[1200:1300] = 0.07
    tags = [("team", 1, [(1500, 1600), (2000, 2100)]), ("team", 1, [(1200, 1300)])]
    draws = tag_stats.team_draws(team, touch, tags, graded=scene()[0], draws=300, seed=(17, 0))
    assert set(np.round(draws, 12)) == {0.0, round(2 / 3, 12), 1.0}, "draws are {}".format(sorted(set(draws)))


def test_tag_stats_per_code_is_aggregates():
    tagged = [{"code": "build up", "start_frame": 1000, "end_frame": 1100, "tags": {"Team": "Barcelona"}},
              {"code": "BUILD UP", "start_frame": 1200, "end_frame": 1300, "tags": {"Team": "Atletic Club"}},
              {"code": "SET PIECES", "start_frame": 1500, "end_frame": 1700, "tags": {}},
              {"code": "OTHER", "start_frame": 1100, "end_frame": 1200, "tags": {}},
              {"code": "LONG BALL", "start_frame": 1550, "end_frame": 1700, "tags": {"Team": "Barcelona"}}]
    inputs = type("Stub", (), {})()
    inputs.smart_tagging = tagged
    inputs.graded = list(BLOCKS)
    _, touch, _ = team_scene()
    touch = touch.copy()
    touch[1550:1600] = 0.93                     # a home episode under LONG BALL (cut at the block's end, 1600)
    rows = tag_stats.per_code(inputs, touch)
    by = {row["code"]: row for row in rows}
    assert set(by) == {"BUILD UP", "LONG BALL"}, "only the team codes are listed: {}".format(sorted(by))
    assert by["BUILD UP"]["frames"] == 200 and by["BUILD UP"]["share"] == 150 / 200, by["BUILD UP"]
    assert by["LONG BALL"]["frames"] == 50 and by["LONG BALL"]["share"] == 1.0, \
        "the tag is cut at the block's end: {}".format(by["LONG BALL"])
    assert all(set(row) == {"code", "frames", "share"} for row in rows), "rows hold code, frames and share only"


def test_tag_stats_rounds_with_the_trainers_helper():
    assert tag_stats.rounded_towards_half(0.2999) == towards_half(0.2999) == 0.3
    assert tag_stats.rounded_towards_half(0.7351) == 0.73, "above half it rounds down"
    assert tag_stats.rounded_towards_half(0.2951) == 0.30, "below half it rounds up"


def test_tag_stats_refuses_bad_arguments():
    team, touch, tags = team_scene()
    graded, missing = scene()
    expect_error(ValueError, lambda: tag_stats.team_agreement(team, touch, tags, graded=graded, ball_missing=missing,
                                                              draws=0), "draws")
    expect_error(ValueError, lambda: tag_stats.team_agreement(team[:-1], touch, tags, graded=graded,
                                                              ball_missing=missing), "one value per frame")


def test_tag_stats_prints_aggregates_only():
    team, touch, tags = team_scene()
    graded, missing = scene(ball_missing=[(1000, 1500)])
    team_out = tag_stats.team_agreement(team, touch, tags, graded=graded, ball_missing=missing, draws=50)
    rows = [{"code": "BUILD UP", "frames": 200, "share": 0.75}]
    text = "\n".join(tag_stats.render(team_out, rows))
    numbers = [token for token in text.replace("\n", " ").replace(",", " ").split() if token.isdigit()]
    scene_frames = {str(frame) for frame in range(1000, N)}
    assert not scene_frames & set(numbers), "a frame of the scene is printed: {}".format(scene_frames & set(numbers))
    assert "BUILD UP" in text and "c_team" in text and "c_dead" not in text, "the figures are printed"
