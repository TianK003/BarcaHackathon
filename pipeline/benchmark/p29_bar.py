"""p29_bar: the bar of P29 to P31 as a script, declared before any run and applied to the benchmark's reports.

    python pipeline/benchmark/p29_bar.py pick --reference REF STEM [STEM ...]   # a lane's pick: STEMs in tie-break order
    python pipeline/benchmark/p29_bar.py bar --reference REF STEM               # STEM against the bar; exit 0 or 2
    python pipeline/benchmark/p29_bar.py guard --reference REF STEM             # the guard alone; exit 0 or 2

The bar (generic profile, both truths, reports of pipeline/benchmark/run.py): the mean of the two truths' pooled
spellF1 at least GAIN above the reference's, the sign-flip p of that mean difference at most P_MAX, and neither truth
more than LOSS below the reference. The guard is the last condition alone. The tie band of `pick` is choose.TIE_BAND.
Every comparison is on the unrounded numbers. Nothing here reads a match.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

from pipeline.benchmark import choose, run  # noqa: E402

GAIN = 0.005
P_MAX = 0.05
LOSS = 0.005
PROFILE = "generic"


def passes_guard(versus):
    """True if neither truth's pooled delta against the reference is below -LOSS (`versus`, as choose.choose's)."""
    return all(versus[truth]["pooled_delta"] >= -LOSS for truth in run.TRUTHS)


def passes_bar(versus):
    """True if the mean of the two truths gains at least GAIN with a sign-flip p of at most P_MAX and the guard holds."""
    mean = versus["mean"]
    return mean["pooled_delta"] >= GAIN and mean["p_sign_flip"] <= P_MAX and passes_guard(versus)


def pick_from(means, order, band=choose.TIE_BAND):
    """Of the stems of `order` whose mean ({stem: mean of the two truths' spellF1}) is within `band` of the highest
    mean, the first in `order`."""
    top = max(means[stem] for stem in order)
    return next(stem for stem in order if means[stem] >= top - band)


def _choice(stems, reference, data_dir):
    return choose.choose(stems, reference, profile=PROFILE, band=choose.TIE_BAND, objective="mean", data_dir=data_dir)


def pick(stems, reference, data_dir=None):
    """The stem the rule gives among `stems` (in tie-break order) against `reference`; prints one line."""
    choice = _choice(stems, reference, data_dir)
    means = {stem: choice["rows"][stem]["mean"]["spell"] for stem in stems}
    picked = pick_from(means, list(stems))
    print("picked    {}".format(picked))
    print("          means {}, reference {} {:.4f}, band {}".format(
        ", ".join("{} {:.4f}".format(stem, means[stem]) for stem in stems), reference,
        choice["rows"][reference]["mean"]["spell"], choose.TIE_BAND))
    return picked


def _line(label, stem, reference, versus, ok, word):
    print("{:9s} {} against {}: mean {:+.4f} (p {:.4f}, wins {} of {}), raw {:+.4f}, coarse {:+.4f}: {}".format(
        label, stem, reference, versus["mean"]["pooled_delta"], versus["mean"]["p_sign_flip"], versus["mean"]["wins"],
        versus["mean"]["n"], versus["raw"]["pooled_delta"], versus["coarse"]["pooled_delta"], word if ok else "FAIL"))


def bar(stem, reference, data_dir=None):
    """True if `stem` clears the bar against `reference`; prints one line with the numbers used."""
    versus = _choice([stem], reference, data_dir)["versus"]
    ok = passes_bar(versus)
    _line("bar", stem, reference, versus, ok, "pass")
    return ok


def guard(stem, reference, data_dir=None):
    """True if neither truth of `stem` is more than LOSS below `reference`; prints one line with the numbers used."""
    versus = _choice([stem], reference, data_dir)["versus"]
    ok = passes_guard(versus)
    _line("guard", stem, reference, versus, ok, "kept")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply the bar of P29 to P31 to the benchmark's reports.")
    parser.add_argument("command", choices=("pick", "bar", "guard"))
    parser.add_argument("stems", nargs="+", help="pick: the stems in tie-break order; bar, guard: one stem")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--data-dir")
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    try:
        if args.command == "pick":
            pick(args.stems, args.reference, args.data_dir)
            return 0
        if len(args.stems) != 1:
            parser.error("{} takes one stem; got {}".format(args.command, args.stems))
        check = bar if args.command == "bar" else guard
        return 0 if check(args.stems[0], args.reference, args.data_dir) else 2
    except (choose.ChooseError, ValueError, OSError, KeyError) as error:
        print("FAILED    {!r}".format(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
