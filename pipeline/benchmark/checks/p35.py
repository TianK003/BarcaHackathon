"""Checks of P35: the hmm decoder's per-frame team cost (test_p35_decoder_*, Task 13) and the turnover hazard's
trainer, source and configs (test_p35_model_*, Task 14).

The decoder's own checks (the brute force of the path and of the posterior, the flat cost against today's loop, the
refusals, the config cross-check, the speed) are pipeline/selftest.py's test_hmm_hazard_*. Here: what diagnose (A19)
does with a config whose decoder reads a hazard. The real turnover_hazard source is Task 14's, so these checks lend the
evidence REGISTRY a stand-in of that kind for their duration (stand_in_hazard) and give the REGISTRY back as it was. No
check reads data.
"""

import contextlib
import tempfile

import numpy as np

from pipeline import evidence
from pipeline.benchmark import diagnose
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark import truth as truths
from pipeline.benchmark.checks.common import synthetic_matches
from pipeline.config import load_config
from pipeline.contracts import A, D, H, Block, BlockContext, Evidence, RunContext, no_opinion
from pipeline.decode import hmm
from pipeline.selftest import quiet, write_text

# Params
HAZARD_KIND = "turnover_hazard"
# A config with a team source, an alive source, a hazard source and the hmm reading it; {scale} is a decoder line.
HAZARD_CONFIG = """
scope = "halves"

[[evidence]]
kind = "nearest_player"

[[evidence]]
kind = "turnover_hazard"
name = "hazard"

[[evidence]]
kind = "ball_presence"

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 8
min_possession = 3
min_dead = 3
hazard_source = "hazard"
{scale}"""
# The same config without the hazard: what scale 0 must reproduce.
PLAIN_CONFIG = HAZARD_CONFIG.replace('[[evidence]]\nkind = "turnover_hazard"\nname = "hazard"\n\n', "").replace(
    'hazard_source = "hazard"\n{scale}', "")


def stand_in_turnover_hazard(run):
    """A stand-in of Task 14's source: a hazard z of made-up values (often near 0 or 1, NaN on 40% of the frames, a
    function of the match's length only, so the same on every call) and no `alive`."""
    n_frames = run.inputs.n_frames
    rng = np.random.default_rng([35, n_frames])
    home = np.where(rng.random(n_frames) < 0.4, np.nan, rng.beta(0.3, 0.3, n_frames))
    return Evidence(alive=no_opinion(n_frames), home=home, name=HAZARD_KIND)


@contextlib.contextmanager
def stand_in_hazard():
    """evidence.REGISTRY with the stand-in as "turnover_hazard" for the duration, and as it was afterwards (the real
    source, once Task 14 is merged, included)."""
    had = HAZARD_KIND in evidence.REGISTRY
    real = evidence.REGISTRY.get(HAZARD_KIND)
    evidence.REGISTRY[HAZARD_KIND] = stand_in_turnover_hazard
    try:
        yield
    finally:
        if had:
            evidence.REGISTRY[HAZARD_KIND] = real
        else:
            evidence.REGISTRY.pop(HAZARD_KIND, None)


def posterior_tally(config, matches):
    """{"alive": (n, mean p), "home": (n, mean p)}: diagnose's posterior measures worked out here from run.predict's
    sources and hmm.decode's own posterior (posterior = true, the config's [decoder] as the pipeline passes it), on the
    clean profile: P(H) + P(A) on every graded frame, and P(H) / (P(H) + P(A)) on the raw truth's alive frames."""
    alive, home = [], []
    for match_id in sorted(matches):
        match = matches[match_id]
        raw, _ = truths.truth_codes(match)
        inputs = match_inputs.MatchInputs(match, benchmark_run.make_atd(match, "clean", 0, None), profile="clean",
                                          seed=0)
        _, sources = benchmark_run.predict(config, inputs)
        for half, start, stop in match.graded:
            block = BlockContext(run=RunContext(inputs=inputs, seed=config.seed), block=Block(half, start, stop),
                                 evidence={name: source.window(start, stop) for name, source in sources.items()})
            posterior = hmm.decode(block, posterior=True, **config.decoder.params).posterior
            in_play = posterior[:, H] + posterior[:, A]
            defined = (raw[start:stop] != D) & (in_play > 0.0)
            alive.append(np.clip(in_play, 0.0, 1.0))
            home.append(np.clip(posterior[defined, H] / in_play[defined], 0.0, 1.0))
    alive, home = np.concatenate(alive), np.concatenate(home)
    return {"alive": (len(alive), float(alive.mean())), "home": (len(home), float(home.mean()))}


def test_p35_decoder_diagnose_reads_the_hazard():
    """diagnose on a config whose decoder reads a hazard: the hazard source is in no calibration entry and no speaker
    key, the posterior is the decoder's own with the hazard (an independent tally through hmm.decode), at hazard_scale
    0 the ladder and the calibration are the same config's without the hazard to the last bit, and at scale 1 the
    hazard moves the posterior. posterior_parameters passes hazard_source and hazard_scale on (1.0 when left out with a
    source). The REGISTRY is as it was afterwards."""
    before = dict(evidence.REGISTRY)
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root, stand_in_hazard():
        reports, configs = {}, {}
        for stem, text in (("p35_plain", PLAIN_CONFIG), ("p35_s000", HAZARD_CONFIG.format(scale="hazard_scale = 0\n")),
                           ("p35_s100", HAZARD_CONFIG.format(scale=""))):
            path = write_text(root, stem + ".toml", text)
            configs[stem] = load_config(path)
            reports[stem] = diagnose.diagnose(path, profile="clean", matches=matches, data_dir=root, log=quiet)
        assert "hazard" not in PLAIN_CONFIG and "{scale}" not in PLAIN_CONFIG, PLAIN_CONFIG
        expected = {"switch_team": 4, "switch_dead": 8, "min_possession": 3, "min_dead": 3, "eps": 1e-6,
                    "team_neutral_dead": False}
        assert diagnose.posterior_parameters(configs["p35_plain"]) == dict(expected, hazard_source=None,
                                                                          hazard_scale=None)
        assert diagnose.posterior_parameters(configs["p35_s000"]) == dict(expected, hazard_source="hazard",
                                                                         hazard_scale=0)
        assert diagnose.posterior_parameters(configs["p35_s100"]) == dict(expected, hazard_source="hazard",
                                                                         hazard_scale=1.0)
        tallies = {stem: posterior_tally(configs[stem], matches) for stem in reports}
        for stem in ("p35_s000", "p35_s100"):
            block = reports[stem]["calibration"]
            assert set(block) == {"nearest_player", "ball_presence", "posterior"}, (stem, sorted(block))
            assert set(block["posterior"]["alive_by_team_sources"]) <= {"nearest_player", "none"}, (
                stem, sorted(block["posterior"]["alive_by_team_sources"]))
            for measure in ("alive", "home"):
                figures = (block["posterior"]["alive"]["all"] if measure == "alive" else block["posterior"]["home"])
                n, mean_p = tallies[stem][measure]
                # the report's floats are rounded to the runner's DECIMALS; the frames are the same, in the same order
                assert figures["n"] == n > 100 and figures["mean_p"] == round(mean_p, benchmark_run.DECIMALS), (
                    stem, measure, figures["n"], figures["mean_p"], tallies[stem][measure])
        plain, zero = reports["p35_plain"], reports["p35_s000"]
        assert zero["ladder"] == plain["ladder"] and zero["calibration"] == plain["calibration"], \
            "hazard_scale 0 is not the config without the hazard"
        assert tallies["p35_s000"] == tallies["p35_plain"], "hazard_scale 0 moved the posterior"
        assert all(abs(tallies["p35_s100"][measure][1] - tallies["p35_plain"][measure][1]) > 1e-9
                   for measure in ("alive", "home")), ("the hazard did not move the posterior", tallies)
    assert evidence.REGISTRY == before, "the stand-in was left in the REGISTRY"
