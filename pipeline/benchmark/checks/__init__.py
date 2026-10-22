"""The benchmark's self-checks, one module per topic. pipeline/benchmark/selftest.py is the harness that runs them.

A check is a function named test_* that takes no arguments and raises if what it checks is wrong. It belongs in the
module of its topic:

    sources.py   the DFL registry and the download
    dfl.py       the DFL parsers and the cache of parsed matches (synthetic, and the downloaded ones with --dfl)
    harness.py   the harness itself: what it collects, how it reports NOT RUN, the console set-up
    common.py    what several modules share: the synthetic DFL match, real_match, damaged, NotRun (no checks of its own)

A new topic gets a new module here, listed in MODULES (the harness refuses a module that is not listed, which would
otherwise run nothing). The harness collects the test_* functions that each module in MODULES defines itself (not
those it imports), and runs them sorted by name across all modules, which is the order the single file always ran in;
the name decides the rest: test_dfl_* needs the downloaded DFL files and runs only with --dfl, test_data_* needs the
real match and runs only with --data. A name used twice is refused, since the harness could not tell the two apart.

A check that cannot run on this machine (an optional dependency is missing) raises common.NotRun(reason): the
harness prints NOT RUN with the reason and counts it apart from the passes. It never returns quietly, which would read
as a pass of something that was not compared.

Import with absolute imports (from pipeline.benchmark import dfl): a check module named like a production module
cannot then shadow it.
"""

from pipeline.benchmark.checks import dfl, harness, sources

MODULES = (sources, dfl, harness)
