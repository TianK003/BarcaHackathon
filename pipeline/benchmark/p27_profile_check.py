"""p27_profile_check: is the generic profile what its spec declares, on the seven matches and three seeds?

    python pipeline/benchmark/p27_profile_check.py [--seeds 3] [--data-dir D]

The spec (docs/superpowers/specs/2026-11-09-p27-degradation-design.md) has one check of the profile itself on the DFL
runs: the pooled share of the ball's masks is 0.41 +- 0.02 and the pooled share of on-pitch player-frames that are
lost is 0.068 +- 0.01, over the 21 match-seeds. A miss is a bug in the code, to be fixed before any choice is read.
This degrades each match under each seed and sums the counts that degrade.degrade records (no frame is kept or
printed), then says pass or fail and exits 0 or 2. It reads DFL's tracking and nothing of our match.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

from pipeline.benchmark import degrade, dfl, run, sources  # noqa: E402

MASK_BAND = (degrade.GENERIC_MISSING_SHARE, 0.02)
LOST_BAND = (1.0 - degrade.GENERIC_PRESENT_MEAN / degrade.FULL, 0.01)


def measure(seeds=run.DEFAULT_SEEDS, data_dir=None, log=print):
    """{"mask": (missing, rows), "lost": (lost, on_pitch)} summed over the registry's matches and `seeds` seeds."""
    missing = rows = lost = on_pitch = 0
    for match_id in sources.MATCH_IDS:
        match = dfl.load_match(match_id, data_dir=data_dir)
        view = degrade.tracking_view(match)
        shares = []
        for seed in range(seeds):
            _, info = degrade.degrade(view, degrade.GENERIC, None, seed)
            for half in info["ball"]["halves"].values():
                missing += half["missing_frames"]
                rows += half["rows"]
            lost += info["players"]["lost_frames"]
            on_pitch += info["players"]["on_pitch_frames"]
            shares.append(info["mask_share"])
        log("match     {}   mask share per seed {}".format(match_id, " ".join("{:.3f}".format(s) for s in shares)))
        del match, view
    return {"mask": (missing, rows), "lost": (lost, on_pitch)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the generic profile against its declared shares.")
    parser.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS)
    parser.add_argument("--data-dir")
    args = parser.parse_args(argv)
    counts = measure(args.seeds, args.data_dir)
    passed = True
    for name, (target, tolerance) in (("mask", MASK_BAND), ("lost", LOST_BAND)):
        part, whole = counts[name]
        share = part / whole
        ok = abs(share - target) <= tolerance
        passed = passed and ok
        print("check     pooled {} share {:.4f} ({:,} of {:,}); declared {:.4f} +- {}: {}".format(
            name, share, part, whole, target, tolerance, "pass" if ok else "FAIL"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
