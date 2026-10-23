"""The benchmark's self-checks, one module per topic. pipeline/benchmark/selftest.py is the harness that runs them.

A check is a function named test_* that takes no arguments and raises if what it checks is wrong. It belongs in the
module of its topic:

    sources.py   the DFL registry and the download
    dfl.py       the DFL parsers and the cache of parsed matches (synthetic, and the downloaded ones with --dfl)
    hazard.py    the gap-onset hazard: features, logistic fit, AUC and the mask sampler (synthetic arrays only)
    calibration.py
                 the calibration, our own tracking measured for the benchmark: every statistic exact on hand-built
                 objects, a synthetic bundle, the cache, the hazard fit of our ball gaps (rows, AUCs, the go rule, the
                 duration cells), and with --data the figures of A01, A02 and R001 and A08's fit
    degrade.py   the degrader's shared parts: the transplant worked out exactly, the draws, what the gaps profile
                 touches and does not, that no truth reaches it, and that the order of the matches changes nothing
                 (synthetic; with --data our real masks on a downloaded DFL match)
    tracking.py  the tracking profile of the degrader: tracklets and slots (walks, lowest free slot, no overlap, the
                 degenerate case equal to gaps, the lost share on long runs), the wrong-team episodes, the NaN-speed
                 runs, the smoothing, and that the profiles nest (synthetic; with --dfl that no pool of slots runs out
                 on the 7 matches; with --data our real Calibration on a downloaded DFL match)
    fidelity.py  the fidelity report: every gate and check worked out by other code on a synthetic Calibration, the
                 pooling by sums, the file and the exit code (synthetic only)
    truth.py     the benchmark's truth from the DFL flags: codes, summary, line timing, nearest team, the coarse
                 truth and the cost of a lag (synthetic, and the downloaded matches with --dfl)
    inputs.py    a DFL match served to the pipeline as MatchInputs and clean_atd: contract, no truth (synthetic, and
                 a downloaded match with --dfl)
    score.py     HalfScore, pooled and per_match against the official pm.evaluate, predict against pipeline.run.run,
                 the yardstick configs, and the match-level statistics (t_interval, sign_flip_p, compare)
                 (synthetic, and a downloaded match with --dfl)
    run.py       the benchmark runner: its reports, the reference and the direction of a comparison, the same bytes in
                 another process, what it refuses, and the oracle, which must score 1.0 on every real match (synthetic,
                 and the downloaded matches with --dfl)
    harness.py   the harness itself: what it collects, how it reports NOT RUN, the console set-up
    common.py    what several modules share: the synthetic DFL match, real_match, damaged, NotRun (no checks of its own)

A new topic gets a new module here, listed in MODULES. The harness refuses a module of this folder that MODULES does
not list (it would run nothing) and an entry of MODULES that is not one of them (a production module of the same name,
say, which would hide that), comparing whole names. It collects the test_* functions that each module in MODULES
defines itself (not those it imports), and runs them sorted by name across all modules, which is the order the single
file always ran in; the name decides the rest: test_dfl_* needs the downloaded DFL files and runs only with --dfl,
test_data_* needs the real match and runs only with --data. A name used twice is refused, since the harness could not
tell the two apart.

A check that cannot run on this machine (an optional dependency is missing) raises common.NotRun(reason): the
harness prints NOT RUN with the reason and counts it apart from the passes. It never returns quietly, which would read
as a pass of something that was not compared.

Import with absolute imports (from pipeline.benchmark import dfl): a check module named like a production module
cannot then shadow it.
"""

from pipeline.benchmark.checks import (
    calibration, degrade, dfl, fidelity, harness, hazard, inputs, run, score, sources, tracking,
    truth)

MODULES = (sources, dfl, truth, inputs, score, run, hazard, calibration, degrade, tracking, fidelity, harness)
