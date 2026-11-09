"""p27_choose: apply P27's declared rules to the benchmark's reports and say what the next stage runs.

    python pipeline/benchmark/p27_choose.py a                 # Stage A's 192 reports -> the touch rule
    python pipeline/benchmark/p27_choose.py c                 # Stage C's 144 reports -> the decoder and the shape weight
    python pipeline/benchmark/p27_choose.py d                 # Stage D's groups -> what changes, group by group
    python pipeline/benchmark/p27_choose.py bar STEM          # the final bar: STEM against the two yardsticks
    python pipeline/benchmark/p27_choose.py guard STEM BASE   # the shift's guard: STEM (shifted) against BASE (not)

The rules are the spec's (docs/superpowers/specs/2026-11-09-p27-degradation-design.md), declared before the runs:
Stages A and C by choose.py's mean rule (the best mean of the two truths' pooled spellF1, a tie band of 0.005, the
larger min(raw, coarse), the stage's tie order) against the reference p27_ref; Stage D by the same rule within each
group, Stage C's choice being one candidate, in the tie order of the entry that first chose the dimension (the
advisor's ruling of 9 Nov; the shape source stays if the config with it is not worse than the config without it). Each command prints aggregates of the DFL benchmark and one last line that starts
with `NEXT`, holding the arguments of the next p27_grid.py call, and writes its selection as JSON under
<data dir>/runs/p27/. Nothing here reads a match.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

import games  # noqa: E402
from pipeline.benchmark import choose, p27_grid, run  # noqa: E402

PROFILE = "generic"
REFERENCE = "p27_ref"
BAND = choose.TIE_BAND
OUT = ("runs", "p27")


def _stage_a():
    settings = [(r, v, k, h) for r in p27_grid.RADII for v in p27_grid.SPEEDS for k in p27_grid.MIN_FRAMES
                for h in p27_grid.HOLDS]
    settings.sort(key=lambda s: (-s[2], s[0], s[1], s[3]))
    return {"p27a_" + p27_grid.touch_stem(s): s for s in settings}


def _stage_c():
    settings = [(t, mp, w) for t in p27_grid.SWITCH_TEAMS for mp in p27_grid.MIN_POSSESSIONS for w in p27_grid.WEIGHTS]
    settings.sort(key=lambda s: (-s[0], -s[1], s[2]))
    return {"p27c_t{:02d}_mp{:03d}_w{:03d}".format(t, mp, int(round(w * 1000))): (t, mp, w) for t, mp, w in settings}


def _write(name, choice, data_dir):
    path = os.path.join(data_dir or games.DATA_DIR, *OUT, name)
    run.write_json(path, run.plain(choice))
    print("written   {}".format(run.shown(path)))


def _show(choice, top):
    for line in choose.render(choice, top):
        print(line)


def _mean(choice, stem):
    return choice["rows"][stem]["mean"]["spell"]


def stage(name, table, data_dir, reference=REFERENCE, top=12):
    choice = choose.choose(list(table), reference, profile=PROFILE, band=BAND, objective="mean", data_dir=data_dir)
    _show(choice, top)
    _write("stage_{}_choice.json".format(name), choice, data_dir)
    return choice


def stage_d(data_dir):
    """Stage D: each group by the rule of Stages A and C, the incumbent (p27d_base) one candidate among the others,
    in the tie order of the entry that first chose it; returns the arguments that differ from Stage C's choice."""
    incumbent = "p27d_base"
    groups = {     # name -> [(stem, the argument that makes it; None for the incumbent)], in the declared tie order
        # R010: larger switch_dead, then larger min_dead.
        "decoder_dead": [(incumbent if (sd, md) == (p27_grid.SWITCH_DEAD, p27_grid.MIN_DEAD)
                          else "p27d_sd{:02d}_md{:02d}".format(sd, md),
                          None if (sd, md) == (p27_grid.SWITCH_DEAD, p27_grid.MIN_DEAD)
                          else "--switch-dead {} --min-dead {}".format(sd, md))
                         for sd in sorted(p27_grid.SWITCH_DEADS, reverse=True)
                         for md in sorted(p27_grid.MIN_DEADS, reverse=True)],
        # R009: the smaller weight.
        "dead_weight": [(incumbent if dw == p27_grid.DEAD_WEIGHT else "p27d_dw{:03d}".format(int(round(dw * 100))),
                         None if dw == p27_grid.DEAD_WEIGHT else "--dead-weight {}".format(dw))
                        for dw in sorted(p27_grid.DEAD_WEIGHTS)],
        # R011: gaps over all.
        "scope": [("p27d_scope_gaps", "--scope gaps"), (incumbent, None)],
    }
    stems = [stem for group in groups.values() for stem, _ in group if stem != incumbent] + ["p27d_noshape"]
    choice = choose.choose([incumbent] + stems, incumbent, profile=PROFILE, band=BAND, objective="mean",
                           data_dir=data_dir)
    _show(choice, None)
    _write("stage_d_choice.json", choice, data_dir)
    rows = choice["rows"]
    arguments, changed = [], 0
    for name, group in groups.items():
        order = [stem for stem, _ in group]
        raw = {stem: rows[stem]["raw"]["spell"] for stem in order}
        coarse = {stem: rows[stem]["coarse"]["spell"] for stem in order}
        best, tied, chosen = choose._select_mean(order, raw, coarse, BAND)
        print("group     {:13s} best {} (mean {:.4f}), tied {}, chosen {} ({:+.4f} against the incumbent){}".format(
            name, best, _mean(choice, best), len(tied), chosen, _mean(choice, chosen) - _mean(choice, incumbent),
            "" if chosen == incumbent else ": CHANGED"))
        if chosen != incumbent:
            arguments.append(dict(group)[chosen])
            changed += 1
    base, without = _mean(choice, incumbent), _mean(choice, "p27d_noshape")
    keeps = base >= without
    print("group     shape         with {:.4f}, without {:.4f}: {}".format(
        base, without, "the shape source stays" if keeps else "the shape source is DROPPED"))
    if not keeps:
        arguments.append("--no-shape")
        changed += 1
    # More than one change: the spec has the combined config run as well, so the chain stops for it.
    print("CHANGED_GROUPS {}".format(changed))
    return arguments


def bar(stem, data_dir):
    """R010's bar: stem against nearest_w50 under the raw truth and nearest_w125 under the coarse truth."""
    passed = True
    for yardstick, truth in (("nearest_w50", "raw"), ("nearest_w125", "coarse")):
        choice = choose.choose([stem], yardstick, profile=PROFILE, band=BAND, objective="mean", data_dir=data_dir)
        cell = choice["versus"][truth]
        ok = cell["pooled_delta"] > 0 and cell["p_sign_flip"] <= 0.05
        passed = passed and ok
        print("bar       {} against {} under the {} truth: {:.4f} against {:.4f}, delta {:+.4f}, p {:.4f}, wins {} of "
              "{}: {}".format(stem, yardstick, truth, choice["rows"][stem][truth]["spell"],
                              choice["rows"][yardstick][truth]["spell"], cell["pooled_delta"], cell["p_sign_flip"],
                              cell["wins"], cell["n"], "pass" if ok else "FAIL"))
    return passed


def guard(stem, base, data_dir):
    """The shift's guard: the shifted config's mean above the unshifted one's, neither truth more than the band below."""
    choice = choose.choose([stem], base, profile=PROFILE, band=BAND, objective="mean", data_dir=data_dir)
    versus = choice["versus"]
    ok = versus["mean"]["pooled_delta"] > 0 and all(versus[truth]["pooled_delta"] >= -BAND for truth in run.TRUTHS)
    print("guard     {} against {}: mean {:+.4f} (p {:.4f}, wins {}), raw {:+.4f}, coarse {:+.4f}: {}".format(
        stem, base, versus["mean"]["pooled_delta"], versus["mean"]["p_sign_flip"], versus["mean"]["wins"],
        versus["raw"]["pooled_delta"], versus["coarse"]["pooled_delta"], "kept" if ok else "NOT kept"))
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply P27's declared rules to the benchmark's reports.")
    parser.add_argument("stage", choices=("a", "c", "d", "bar", "guard"))
    parser.add_argument("stems", nargs="*")
    parser.add_argument("--data-dir")
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    try:
        if args.stage == "a":
            table = _stage_a()
            chosen = table[stage("a", table, args.data_dir)["chosen"]]
            print("NEXT --touch {},{},{},{}".format(*chosen))
        elif args.stage == "c":
            table = _stage_c()
            choice = stage("c", table, args.data_dir)
            t, mp, w = table[choice["chosen"]]
            edge = [name for name, value, values in (("switch_team", t, p27_grid.SWITCH_TEAMS),
                                                      ("min_possession", mp, p27_grid.MIN_POSSESSIONS),
                                                      ("weight", w, p27_grid.WEIGHTS)) if value in (values[0], values[-1])]
            if edge:
                print("note      the choice is on the edge of the grid in: {} (reported, not extended)".format(
                    ", ".join(edge)))
            print("NEXT --decoder {},{} --weight {}".format(t, mp, w))
        elif args.stage == "d":
            print("NEXT {}".format(" ".join(stage_d(args.data_dir))))
        elif args.stage == "bar":
            return 0 if bar(args.stems[0], args.data_dir) else 2
        else:
            return 0 if guard(args.stems[0], args.stems[1], args.data_dir) else 2
    except (choose.ChooseError, ValueError, OSError, IndexError, KeyError) as error:
        print("FAILED    {!r}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
