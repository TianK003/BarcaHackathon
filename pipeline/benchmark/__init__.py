"""The P01 validation benchmark: score pipeline choices on matches that are not ours (docs/proposals.md, P01).

The organisers ruled on 22 Oct 2026 that BAR-ATH may not be hand-labelled, not even to validate. So the choices that
matter (decoder, evidence weights, thresholds) are validated on the seven open DFL/IDSSE Bundesliga matches (CC BY 4.0):
real tracking with the ball's own alive/dead and possession flags, to be degraded until it looks like our tracking.

    python pipeline/benchmark/sources.py              # download the 21 DFL files (2.6 GB) into data/external/dfl/,
                                                      # then check every size and md5
    python pipeline/benchmark/sources.py --verify     # only check what is already there
    python pipeline/benchmark/run.py CONFIG [CONFIG ...]    # score configs on the seven matches (README.md)
    python pipeline/benchmark/fidelity.py             # how far the degraded profile looks like our match
    python pipeline/benchmark/selftest.py             # synthetic checks, seconds
    python pipeline/benchmark/selftest.py --dfl       # + the downloaded files
    python pipeline/benchmark/selftest.py --data      # + our own match (aggregates only)

This file grows with the benchmark: sources.py is the registry and the download; dfl.py parses a match and keeps it
parsed under data/cache/dfl/ (dfl.load_match is the one way every later stage gets a match); hazard.py is the maths of
where our tracker loses the ball (per-frame features, a logistic fit, a sampler of ball gaps) and reads no data;
calibration.py measures our own tracking (ball_stats and player_stats, the two functions that measure BAR-ATH here and a
degraded benchmark match later, and a Calibration of BAR-ATH that load() measures once and keeps under
data/cache/benchmark/calibration/; docs/results.md A07);
truth.py turns a match's flags into the scorer's H/A/D/X codes and reads off them what A06 measured; inputs.py serves
a match to the pipeline as an Inputs (MatchInputs, clean_atd), with no truth in it; run.py runs a config on it through
the pipeline's own functions (predict) and is the benchmark's command (evaluate_configs: every config on every match,
scored, compared with the first, with one JSON report each); score.py scores the result half by half and pools the
counts into the official scorer's number (HalfScore, score_halves, pooled, per_match) and holds the match-level
statistics (t_interval, sign_flip_p, compare, kendall_tau); configs/ holds the yardsticks, the organisers' four baselines
as per-half configs; degrade.py makes the tracking of a match look like ours from a view of it that has no truth in it
(TrackingView, rng_for, the `gaps` profile: our own missing-ball masks, transplanted) and fidelity.py measures how far it
got, with calibration.py's own statistics functions (gates that must pass, warnings that are discussed). README.md here
says how to download the data, run a config and read what comes out.

The self-checks live in checks/, one module per topic, and selftest.py is the harness that collects and runs them (to
add a check, see checks/__init__.py).
"""
