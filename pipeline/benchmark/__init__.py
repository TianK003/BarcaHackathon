"""The P01 validation benchmark: score pipeline choices on matches that are not ours (docs/proposals.md, P01).

The organisers ruled on 22 Oct 2026 that BAR-ATH may not be hand-labelled, not even to validate. So the choices that
matter (decoder, evidence weights, thresholds) are validated on the seven open DFL/IDSSE Bundesliga matches (CC BY 4.0):
real tracking with the ball's own alive/dead and possession flags, to be degraded until it looks like our tracking.

    python pipeline/benchmark/sources.py              # download the 21 DFL files (2.6 GB) into data/external/dfl/,
                                                      # then check every size and md5
    python pipeline/benchmark/sources.py --verify     # only check what is already there
    python pipeline/benchmark/selftest.py             # synthetic checks, seconds
    python pipeline/benchmark/selftest.py --dfl       # + the downloaded files

This file grows with the benchmark: sources.py is the registry and the download; dfl.py parses a match and keeps it
parsed under data/cache/dfl/ (dfl.load_match is the one way every later stage gets a match); hazard.py is the maths of
where our tracker loses the ball (per-frame features, a logistic fit, a sampler of ball gaps) and reads no data;
truth.py turns a match's flags into the scorer's H/A/D/X codes and reads off them what A06 measured; the scorer and the
degrader land in their own modules next to it.

The self-checks live in checks/, one module per topic, and selftest.py is the harness that collects and runs them (to
add a check, see checks/__init__.py).
"""
