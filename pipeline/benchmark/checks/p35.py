"""Checks of P35: the hmm decoder's per-frame team cost (test_p35_decoder_*, Task 13) and the turnover hazard's
trainer, source and configs (test_p35_model_*, Task 14).

The decoder's own checks (the brute force, the flat cost against today's loop, the refusals, the config cross-check, the
speed) are pipeline/selftest.py's test_hmm_hazard_*. Here: what diagnose (A19) does with a config whose decoder reads a
hazard. The real turnover_hazard source is Task 14's, so these checks lend the evidence REGISTRY a stand-in of that kind
for their duration (stand_in_hazard) and give the REGISTRY back as it was. No check reads data.
"""

import contextlib
import tempfile

from pipeline import evidence
from pipeline.benchmark import diagnose
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import loading
from pipeline.config import load_config
from pipeline.contracts import Evidence, no_opinion
from pipeline.selftest import expect_error, quiet, write_text

# Params
HAZARD_KIND = "turnover_hazard"
# A config with a team source, an alive source, a hazard source and the hmm reading it.
HAZARD_CONFIG = """
scope = "halves"

[[evidence]]
kind = "nearest_player"

[[evidence]]
kind = "ball_presence"

[[evidence]]
kind = "turnover_hazard"
name = "hazard"

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 8
min_possession = 3
min_dead = 3
hazard_source = "hazard"
"""


def stand_in_turnover_hazard(run):
    """A stand-in of Task 14's source: no opinion anywhere. Only its kind and its name matter to these checks."""
    n_frames = run.inputs.n_frames
    return Evidence(alive=no_opinion(n_frames), home=no_opinion(n_frames), name=HAZARD_KIND)


@contextlib.contextmanager
def stand_in_hazard():
    """evidence.REGISTRY with a stand-in "turnover_hazard" for the duration (unless it has the real one), as it was
    afterwards."""
    real = evidence.REGISTRY.get(HAZARD_KIND)
    if real is None:
        evidence.REGISTRY[HAZARD_KIND] = stand_in_turnover_hazard
    try:
        yield
    finally:
        if real is None:
            evidence.REGISTRY.pop(HAZARD_KIND, None)


def test_p35_decoder_diagnose_refuses_a_hazard():
    """diagnose's posterior block refuses a config whose decoder reads a hazard (hmm.marginals takes no per-frame team
    cost), with a BenchmarkError naming the config and hazard_source, before any match is loaded; posterior_parameters
    says the same of the loaded config. The REGISTRY is as it was afterwards."""
    def serve(match_id):
        raise AssertionError("a match was loaded for a config that is refused: " + match_id)

    before = dict(evidence.REGISTRY)
    with tempfile.TemporaryDirectory() as root, loading(serve), stand_in_hazard():
        path = write_text(root, "p35_hazard.toml", HAZARD_CONFIG)
        expect_error(benchmark_run.BenchmarkError, lambda: diagnose.posterior_parameters(load_config(path)),
                     "p35_hazard", "hazard_source", "'hazard'")
        expect_error(benchmark_run.BenchmarkError,
                     lambda: diagnose.diagnose(path, profile="clean", data_dir=root, log=quiet),
                     "p35_hazard", "hazard_source", "'hazard'")
        # the same config without the hazard is diagnosed as before: its parameters are the decoder's
        plain = HAZARD_CONFIG.replace('[[evidence]]\nkind = "turnover_hazard"\nname = "hazard"\n\n', "").replace(
            'hazard_source = "hazard"\n', "")
        assert "hazard" not in plain
        assert diagnose.posterior_parameters(load_config(write_text(root, "p35_plain.toml", plain))) == {
            "switch_team": 4, "switch_dead": 8, "min_possession": 3, "min_dead": 3, "eps": 1e-6,
            "team_neutral_dead": False, "hazard_source": None, "hazard_scale": None}
    assert evidence.REGISTRY == before, "the stand-in was left in the REGISTRY"
