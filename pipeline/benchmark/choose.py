"""choose: apply a grid's declared selection rule to the benchmark's written reports, and estimate it out of sample.

    python pipeline/benchmark/choose.py --reference main CONFIG_STEM [CONFIG_STEM ...] [--profile tracking]
                                        [--band 0.005] [--metric spell|macro] [--data-dir D] [--out PATH]

    choice = choose(stems, reference="main")      # {"rows", "tied", "chosen", "versus", "lomo", ...}

The rule (R008 to R013, docs/results.md). The objective is pooled spellF1 under the raw truth; every config within the
tie band (0.005) of the best is tied; among the tied, the best pooled spellF1 under the coarse truth wins; a tie that is
left goes to the config named first, so the stems are given in the declared tie order. The reference (`main`, say) is
the config every delta is taken against, and it is never a candidate unless it is also named among the stems.

Leave-one-match-out (lomo), as R008 to R011 computed it by hand from the printed per-match scores: for each match, the
selection is repeated on the other matches, with each config's mean per-match spellF1 over them in place of the pooled
score, and the chosen config's score on the held-out match is compared with the reference's on it. Two rules: the raw
objective alone (the best six-match raw mean, the tie order for an exact tie) and the whole rule (the band, then the
coarse mean, then the tie order). Each gives, per truth, the held-out deltas, their mean with its 90% t interval, the
exact sign-flip p and the wins, and which config each fold picked.

It reads only the reports that pipeline/benchmark/run.py wrote (<data dir>/runs/benchmark/<profile>/<stem>.json and
<stem>.coarse.json, all seven matches, the default seeds) and computes everything from their halves with score.py, so
its numbers are the runner's. A report is refused if its config file has changed since it was written (the report
holds the config's text), or if two reports differ in their matches, seeds, profile or Calibration: a selection over
reports of different runs would compare different things. The reports hold DFL's counts and no frame of BAR-ATH.

`metric="macro"` (--metric macro, P12) applies the same rule to pooled macroF1 in place of spellF1: the raw macroF1
chooses, the coarse macroF1 breaks a tie, the deltas, p and wins (rows, versus and lomo) are macroF1's, and each
match's score in the lomo is its macroF1. The band is in macroF1's units, so a grid judged on it names its band.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import games  # noqa: E402
from pipeline.benchmark import run, score  # noqa: E402

# Params
TIE_BAND = 0.005                 # spellF1: the configs within this of the best raw score are tied (R008 to R013)
SPELL = "spell_f1_mean"
METRICS = {"spell": SPELL, "macro": "macro_f1"}      # the objective's name -> the scalar of score.pooled that it is
SAME = ("matches", "seeds", "profile", "calibration")    # what every report of one selection must share


class ChooseError(ValueError):
    """A selection that cannot be made from the reports as they are; the message says which report and why."""


def load(stem, truth, profile=run.DEFAULT_PROFILE, data_dir=None, match_ids=None, seeds=None):
    """The report of `stem` under `truth`, checked against its config file; ChooseError if absent or stale.

    `match_ids` and `seeds` name a report of another run than the full one, as run.report_path does.
    """
    path = run.report_path(data_dir or games.DATA_DIR, profile, stem, truth, match_ids, seeds)
    if not os.path.exists(path):
        raise ChooseError("{}: no report under the {} truth at {}; run it with pipeline/benchmark/run.py --profile {} "
                          "--truth both".format(stem, truth, run.shown(path), profile))
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    config = os.path.join(ROOT, report["config"]["path"])
    if not os.path.exists(config):
        raise ChooseError("{}: the report was written for {}, which is not there any more".format(
            stem, report["config"]["path"]))
    with open(config, encoding="utf-8", newline="") as stream:
        text = stream.read().replace("\r\n", "\n")
    if text != report["config"]["text"]:
        raise ChooseError("{}: {} has changed since its report was written; run it again".format(
            stem, report["config"]["path"]))
    return report


def halves(report):
    """The report's HalfScores, as score.py takes them."""
    return [score.HalfScore(match_id=half["match_id"], half=half["half"], seed=half["seed"], n_gt=half["n_gt"],
                            n_pred=half["n_pred"], tp=tuple(half["tp"]),
                            confusion=tuple(tuple(row) for row in half["confusion"])) for half in report["halves"]]


def choose(stems, reference, profile=run.DEFAULT_PROFILE, band=TIE_BAND, data_dir=None, match_ids=None, seeds=None,
           metric="spell"):
    """The rule applied to `stems` (in the declared tie order) against `reference`; see the module docstring.

    Returns {"reference", "profile", "matches", "seeds", "rows", "best", "tied", "chosen", "versus", "lomo"}: one row
    per stem (and the reference) with its pooled scores under both truths, its alive/dead F1, possessions per graded
    minute, D share and per-match score (of `metric`); the best raw stem, the tied stems and the chosen one; the
    chosen config against the reference under each truth (score.compare); and the two leave-one-match-out rules.
    `metric` is the objective, "spell" (pooled spellF1, the default) or "macro" (pooled macroF1): it picks the rows'
    score that selects, what score.compare compares and what the lomo averages; every other field is unchanged.
    """
    if metric not in METRICS:
        raise ChooseError("metric must be one of {}; got {!r}".format(sorted(METRICS), metric))
    stems = list(stems)
    if not stems or len(set(stems)) < len(stems):
        raise ChooseError("name one or more configs, each once; got {}".format(stems))
    names = [reference] + [stem for stem in stems if stem != reference]
    reports = {(stem, truth): load(stem, truth, profile, data_dir, match_ids, seeds)
               for stem in names for truth in run.TRUTHS}
    first = reports[(reference, "raw")]
    for (stem, truth), report in reports.items():
        for key in SAME:
            if report[key] != first[key]:
                raise ChooseError("{} ({} truth) differs from {} in its {}; the reports are of different runs".format(
                    stem, truth, reference, key))
    scores = {key: halves(report) for key, report in reports.items()}

    rows = {}
    for stem in names:
        row = {}
        for truth in run.TRUTHS:
            pooled = score.pooled(scores[(stem, truth)])
            row[truth] = {"spell": pooled[SPELL], "macro": pooled["macro_f1"],
                          "per_match": {match_id: entry[METRICS[metric]]
                                        for match_id, entry in score.per_match(scores[(stem, truth)]).items()}}
        raw = reports[(stem, "raw")]
        row["alive_dead"] = raw["pooled"]["alive_dead_f1"]
        row["per_minute"] = raw["possessions_per_graded_minute"]["predicted"]
        row["d_share"] = raw["labels"]["predicted"]["D"]
        rows[stem] = row

    best, tied, chosen = _select(stems, {stem: rows[stem]["raw"][metric] for stem in stems},
                                 {stem: rows[stem]["coarse"][metric] for stem in stems}, band)
    versus = {truth: score.compare(scores[(reference, truth)], scores[(chosen, truth)], metric=METRICS[metric])
              for truth in run.TRUTHS}
    for truth in run.TRUTHS:
        for stem in names:
            if stem != reference:
                fields = score.compare(scores[(reference, truth)], scores[(stem, truth)], metric=METRICS[metric])
                rows[stem][truth].update(delta=fields["pooled_delta"], p=fields["p_sign_flip"], wins=fields["wins"])
    return {"reference": reference, "metric": metric, "profile": first["profile"],
            "matches": [m["id"] for m in first["matches"]], "seeds": first["seeds"], "band": band, "rows": rows,
            "order": stems, "best": best, "tied": tied, "chosen": chosen, "versus": versus,
            "lomo": lomo(stems, reference, rows, band)}


def _select(stems, raw, coarse, band):
    """(best, tied, chosen): the best raw stem, those within `band` of it, and the rule's choice among them."""
    top = max(raw[stem] for stem in stems)
    best = next(stem for stem in stems if raw[stem] == top)
    tied = [stem for stem in stems if raw[stem] >= top - band]
    top_coarse = max(coarse[stem] for stem in tied)
    chosen = next(stem for stem in tied if coarse[stem] == top_coarse)
    return best, tied, chosen


def lomo(stems, reference, rows, band=TIE_BAND):
    """{"objective", "rule"}: the two leave-one-match-out estimates of the selection; see the module docstring."""
    matches = sorted(rows[reference]["raw"]["per_match"])
    out = {}
    for name in ("objective", "rule"):
        picks, deltas = {}, {truth: [] for truth in run.TRUTHS}
        for held in matches:
            kept = [match_id for match_id in matches if match_id != held]

            def mean(stem, truth):
                return sum(rows[stem][truth]["per_match"][match_id] for match_id in kept) / len(kept)

            raw = {stem: mean(stem, "raw") for stem in stems}
            if name == "objective":
                pick = next(stem for stem in stems if raw[stem] == max(raw.values()))
            else:
                pick = _select(stems, raw, {stem: mean(stem, "coarse") for stem in stems}, band)[2]
            picks[held] = pick
            for truth in run.TRUTHS:
                deltas[truth].append(rows[pick][truth]["per_match"][held] - rows[reference][truth]["per_match"][held])
        out[name] = {"picks": picks}
        for truth in run.TRUTHS:
            mean_delta, low, high = score.t_interval(deltas[truth])
            out[name][truth] = {"deltas": dict(zip(matches, deltas[truth])), "mean": mean_delta, "low": low,
                                "high": high, "p": score.sign_flip_p(deltas[truth]),
                                "wins": sum(1 for delta in deltas[truth] if delta > score.FLIP_TOLERANCE),
                                "losses": sum(1 for delta in deltas[truth] if delta < -score.FLIP_TOLERANCE)}
    return out


# -- the printout -----------------------------------------------------------------------------------------------------

def render(choice, rows_shown=None):
    """The lines: the table (every row, or the `rows_shown` best by raw score), the choice, its deltas, the lomo."""
    rows, reference, metric = choice["rows"], choice["reference"], choice["metric"]
    order = sorted(choice["order"], key=lambda stem: -rows[stem]["raw"][metric])
    shown = order if rows_shown is None else order[:rows_shown]
    lines = ["profile {} {}   matches {}   seeds {}   reference {}   metric {}F1   tie band {}".format(
        choice["profile"]["name"], choice["profile"]["version"], len(choice["matches"]),
        ",".join(str(seed) for seed in choice["seeds"]), reference, metric, choice["band"]), "",
        "| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | "
        "D share | raw Δ | p | wins | coarse Δ | p | wins |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for stem in [reference] + [stem for stem in shown if stem != reference]:
        row = rows[stem]
        mark = "**" if stem == choice["chosen"] else ""
        cells = ["{}`{}`{}".format(mark, stem, mark), "{:.4f}".format(row["raw"]["spell"]),
                 "{:.4f}".format(row["coarse"]["spell"]), "{:.4f}".format(row["raw"]["macro"]),
                 "{:.4f}".format(row["coarse"]["macro"]), "{:.4f}".format(row["alive_dead"]),
                 "{:.2f}".format(row["per_minute"]), "{:.1%}".format(row["d_share"])]
        for truth in run.TRUTHS:
            if stem == reference:
                cells += ["", "", ""]
            else:
                cells += ["{:+.4f}".format(row[truth]["delta"]), "{:.4f}".format(row[truth]["p"]),
                          str(row[truth]["wins"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "best raw  {} {:.4f}".format(choice["best"], rows[choice["best"]]["raw"][metric]),
              "tied      {} within {} of it: {}".format(len(choice["tied"]), choice["band"], ", ".join(
                  "{} ({:.4f} raw, {:.4f} coarse)".format(stem, rows[stem]["raw"][metric],
                                                          rows[stem]["coarse"][metric]) for stem in choice["tied"])),
              "chosen    {}".format(choice["chosen"])]
    for truth in run.TRUTHS:
        fields = choice["versus"][truth]
        lines.append("versus    {} {}: pooled {:+.4f}, mean per match {:+.4f} [{:+.4f}, {:+.4f}], p {:.4f}, wins {}, "
                     "losses {}; per match {}".format(
                         reference, truth, fields["pooled_delta"], fields["mean_delta"], fields["low"], fields["high"],
                         fields["p_sign_flip"], fields["wins"], fields["losses"],
                         " ".join("{:+.4f}".format(delta) for delta in fields["match_deltas"].values())))
    for name, label in (("objective", "the raw objective alone"), ("rule", "the whole rule")):
        entry = choice["lomo"][name]
        lines.append("lomo      {}: picks {}".format(label, ", ".join(
            "{} {}".format(held, pick) for held, pick in entry["picks"].items())))
        for truth in run.TRUTHS:
            cell = entry[truth]
            lines.append("          {}: mean {:+.4f} [{:+.4f}, {:+.4f}], p {:.4f}, wins {}, losses {}; per match {}"
                         .format(truth, cell["mean"], cell["low"], cell["high"], cell["p"], cell["wins"],
                                 cell["losses"], " ".join("{:+.4f}".format(d) for d in cell["deltas"].values())))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply the declared selection rule to a grid's benchmark reports.")
    parser.add_argument("stems", nargs="+", metavar="CONFIG_STEM",
                        help="the grid's configs by name, in the declared tie order (the first wins a tie that is left)")
    parser.add_argument("--reference", required=True, help="the config every delta is taken against, e.g. main")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--band", type=float, default=None,
                        help="the tie band (default {} for spellF1; required with --metric macro)".format(TIE_BAND))
    parser.add_argument("--metric", choices=sorted(METRICS), default="spell",
                        help="the objective: pooled spellF1 (default) or macroF1; the band is in its units")
    parser.add_argument("--top", type=int, help="show only this many rows of the table, the best raw first")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--out", help="also write the selection as JSON here")
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)      # the table says Δ, which a Windows console's code page
    if reconfigure is not None:                                 # cannot print; a replaced stdout may not have it
        reconfigure(encoding="utf-8")
    if args.band is None:
        if args.metric == "macro":
            print("FAILED    a macroF1 grid must name its band with --band: the default {} is spellF1's".format(TIE_BAND))
            return 1
        args.band = TIE_BAND
    try:
        choice = choose(args.stems, args.reference, profile=args.profile, band=args.band, metric=args.metric,
                        data_dir=args.data_dir)
    except (ChooseError, ValueError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    for line in render(choice, args.top):
        print(line)
    if args.out:
        run.write_json(args.out, run.plain(choice))
        print("written   {}".format(run.shown(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
