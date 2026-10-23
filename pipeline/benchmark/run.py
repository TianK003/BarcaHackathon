"""Run a pipeline config on a benchmark match: the pipeline's own evidence, decode and refine stages, without the CSV.

    codes, sources = predict(config, inputs)       # inputs is a MatchInputs (inputs.py); codes is (n_frames,) int8

Why it imports pipeline.run's functions instead of copying them. The benchmark is only worth anything if it scores what
the submission pipeline does, so prediction goes through pipeline.run.compute_evidence, make_blocks and decode_blocks,
the same functions pipeline.run.run calls, and a change to them changes the benchmark with it. What predict leaves out
of run() is the output stage: it writes no file. Benchmark code never writes a CSV of labels (a CSV of labels of
BAR-ATH is match data, CLAUDE.md), and a benchmark match has no CSV to write; the scoring reads the codes directly
(score.py) and pipeline.output.summarise describes them.

Seeded as run() seeds: `random` and `np.random` with config.seed before anything else, because library code reaches
for the global generators, and the RunContext carries the same seed for components that make their own generator.
Two calls with one config on one inputs give the same codes, and the same as pipeline.run.run on them
(test_predict_equals_pipeline_run).

The pipeline's modules are imported by their absolute names (`from pipeline import run as pipeline_run`): this file is
called run.py and pipeline/benchmark/inputs.py is called inputs.py, like pipeline/run.py and pipeline/inputs/. Never
`import run`: with this folder on sys.path it would be this file.
"""

import random

import numpy as np

from pipeline import run as pipeline_run
from pipeline.contracts import RunContext


def predict(config, inputs):
    """(codes, sources): the codes the config gives on `inputs`, and the evidence they were decoded from.

    `codes` is the (n_frames,) int8 array of pm's codes that pipeline.run.run would write to its CSV: the decoder and
    the refiners, block by block, and X on every frame outside the graded halves. `sources` is {name: Evidence} over
    the whole match, read-only, the same thing RunResult.evidence holds: a failing source can be named from it, and
    pipeline.evidence.describe takes it. ValueError for scope = "file", which decodes across half time and over frames
    that are not graded (docs/traps.md T11); the benchmark scores each half on its own, so only "halves" means anything
    here, and the one config that needs "file" is the reproduction of the published baselines.
    """
    if config.scope == "file":
        raise ValueError("{}: scope = \"file\" decodes across half time and over ungraded frames, and the benchmark "
                         "scores each graded half on its own; use scope = \"halves\" (only "
                         "pipeline/configs/smooth50_repro.toml needs \"file\", to rebuild a published baseline byte "
                         "for byte)".format(config.stem))
    random.seed(config.seed)
    np.random.seed(config.seed)
    run_context = RunContext(inputs=inputs, seed=config.seed)
    sources = pipeline_run.compute_evidence(config, run_context)
    blocks = pipeline_run.make_blocks(config.scope, inputs)
    codes = pipeline_run.decode_blocks(config, run_context, sources, blocks)
    return codes, sources
