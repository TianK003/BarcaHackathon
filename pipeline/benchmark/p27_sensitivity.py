"""p27_sensitivity: R030's sensitivity to the degradation seeds. Made after the choice; it chooses nothing.

    python pipeline/benchmark/p27_sensitivity.py REFERENCE STEM [STEM ...] [--seeds 6] [--data-dir D]

R030 (docs/results.md, "After the choice") declared one run of six configs at six seeds instead of three, because two
of its choices turned on the edge of the 0.005 tie band. This reads that run's reports,
<data dir>/runs/benchmark/generic/<stem>.s6.json and <stem>.coarse.s6.json, and prints, for each config, the pooled
spellF1 under the raw and the coarse truth and their mean over three sets of seeds: 0 to 2 (the seeds of the choice,
which must reproduce the three-seed reports), 3 to 5 (the only ones that played no part in the selection) and 0 to 5.
Each set is pooled from the reports' per-half counts with score.pooled, the scorer's own pooling. For every config
but the reference it adds the delta of the mean against the reference, per match and with the exact sign-flip p.

It prints aggregates of the DFL benchmark and nothing of our match. It applies no rule and names no winner: the
declared outcome of R030 stands whatever this shows.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

import games  # noqa: E402
from pipeline.benchmark import run, score  # noqa: E402

PROFILE = "generic"
SETS = (("seeds 0-2", (0, 1, 2)), ("seeds 3-5", (3, 4, 5)), ("seeds 0-5", (0, 1, 2, 3, 4, 5)))


def half_scores(document, seeds):
    """The HalfScores of a report's halves whose seed is in `seeds`."""
    return [score.HalfScore(match_id=half["match_id"], half=half["half"], seed=half["seed"], n_gt=half["n_gt"],
                            n_pred=half["n_pred"], tp=tuple(half["tp"]),
                            confusion=tuple(tuple(row) for row in half["confusion"]))
            for half in document["halves"] if half["seed"] in seeds]


def read(stem, truth, seeds, data_dir):
    path = run.report_path(data_dir, PROFILE, stem, truth, None, seeds)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document["seeds"] != list(range(seeds)):
        raise SystemExit("p27_sensitivity: {} holds seeds {}, not 0 to {}".format(path, document["seeds"], seeds - 1))
    return document


def measure(stems, seeds=6, data_dir=None):
    """{stem: {set name: {"raw", "coarse", "mean", "per_match": {match: mean of the two truths}}}}."""
    data_dir = data_dir or games.DATA_DIR
    out = {}
    for stem in stems:
        documents = {truth: read(stem, truth, seeds, data_dir) for truth in run.TRUTHS}
        out[stem] = {}
        for name, members in SETS:
            cell, matches = {}, {}
            for truth in run.TRUTHS:
                scores = half_scores(documents[truth], members)
                cell[truth] = score.pooled(scores)["spell_f1_mean"]
                for match_id, pooled in score.per_match(scores).items():
                    matches.setdefault(match_id, []).append(pooled["spell_f1_mean"])
            cell["mean"] = sum(cell[truth] for truth in run.TRUTHS) / len(run.TRUTHS)
            cell["per_match"] = {match_id: sum(values) / len(values) for match_id, values in sorted(matches.items())}
            out[stem][name] = cell
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="R030's sensitivity to the seeds: a report, never a choice.")
    parser.add_argument("reference")
    parser.add_argument("stems", nargs="+")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--data-dir")
    args = parser.parse_args(argv)
    stems = [args.reference] + args.stems
    result = measure(stems, args.seeds, args.data_dir)
    print("R030, sensitivity to the seeds: made after the choice, outside the spec, chooses nothing.")
    print("profile {}   reference {}   pooled spellF1 (raw / coarse / mean of the two)".format(PROFILE, args.reference))
    print()
    print("{:24s}".format("config") + "".join("   {:^24s}".format(name) for name, _ in SETS))
    for stem in stems:
        print("{:24s}".format(stem) + "".join("   {:.4f} / {:.4f} / {:.4f}".format(
            result[stem][name]["raw"], result[stem][name]["coarse"], result[stem][name]["mean"]) for name, _ in SETS))
    print()
    print("the mean of the two truths against {}: pooled delta, then per match (sign-flip p, wins of 7)".format(
        args.reference))
    for stem in args.stems:
        for name, _ in SETS:
            ours, theirs = result[stem][name], result[args.reference][name]
            deltas = [ours["per_match"][match] - theirs["per_match"][match] for match in ours["per_match"]]
            wins = sum(1 for delta in deltas if delta > score.FLIP_TOLERANCE)
            print("{:24s} {:9s}  pooled {:+.4f} (raw {:+.4f}, coarse {:+.4f})   per match {}   p {:.4f}, wins {}".format(
                stem, name, ours["mean"] - theirs["mean"], ours["raw"] - theirs["raw"],
                ours["coarse"] - theirs["coarse"], " ".join("{:+.4f}".format(delta) for delta in deltas),
                score.sign_flip_p(deltas), wins))
    return 0


if __name__ == "__main__":
    sys.exit(main())
