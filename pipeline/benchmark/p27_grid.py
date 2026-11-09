"""p27_grid: write the configs of P27's re-choice (docs/superpowers/specs/2026-11-09-p27-degradation-design.md).

    python pipeline/benchmark/p27_grid.py ref   --confidences FILE
    python pipeline/benchmark/p27_grid.py a     --confidences FILE
    python pipeline/benchmark/p27_grid.py c     --confidences FILE --touch R,V,K,H
    python pipeline/benchmark/p27_grid.py d     --confidences FILE --touch R,V,K,H --decoder T,MP --weight W
    python pipeline/benchmark/p27_grid.py final --confidences FILE --touch R,V,K,H --decoder T,MP --weight W
                                                [--switch-dead SD] [--min-dead MD] [--dead-weight DW] [--scope S]
                                                [--no-shape] [--shift TEAM,INTO_DEAD,OUT_OF_DEAD] [--name STEM]

Every config is a full TOML of the submission pipeline, written to pipeline/benchmark/configs/p27/, and each command
prints the stems it wrote in the stage's tie order (choose.py gives a remaining tie to the stem named first). FILE is
what `touch_stats.py --grid r013 --profile generic --seeds 3 --out FILE` wrote: the confidence of a setting of the
touch rule is read from it and never typed in. The grids are R013's (Stages A and C) and the spec's Stage D; the
incumbent values are main.toml's as R024 left them. Nothing here reads a match.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, "configs", "p27")

# The incumbent: main.toml as R024 left it, without the shift (the grids are scored before the refiner, as R013's).
TOUCH = (1.5, 6.0, 1, 0.55)              # radius m, max_rel_speed m/s, min_frames, hold
DECODER = (4, 35)                        # switch_team, min_possession
SWITCH_DEAD, MIN_DEAD = 32, 25
DEAD_WEIGHT, SHAPE_WEIGHT, SCOPE, MIN_PROB = 1.0, 0.2, "all", 0.02
# Stage A (R013's): 4 x 3 x 4 x 4 = 192.
RADII, SPEEDS, MIN_FRAMES, HOLDS = (1.5, 2.0, 2.5, 3.0), (6.0, 9.0, 12.0), (1, 2, 3, 5), (0.52, 0.55, 0.60, 0.70)
# Stage C (R013's): 6 x 6 x 4 = 144.
SWITCH_TEAMS, MIN_POSSESSIONS, WEIGHTS = (1, 2, 3, 4, 6, 8), (1, 10, 15, 25, 35, 50), (0.1, 0.2, 0.3, 0.5)
# Stage D (the spec's): the dimensions R009, R010 and R011 chose, one group at a time.
SWITCH_DEADS, MIN_DEADS, DEAD_WEIGHTS, SCOPES = (16, 32, 64), (1, 25), (0.1, 0.25, 0.5, 1.0), ("gaps", "all")


def confidences(path):
    """{(radius, max_rel_speed, min_frames): confidence} of a touch_stats result file."""
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    table = {}
    for entry in result["settings"].values():
        setting = entry["setting"]
        if entry["confidence"] is None:
            raise SystemExit("p27_grid: {} has no confidence for {}".format(path, setting))
        table[(float(setting["radius"]), float(setting["max_rel_speed"]), int(setting["min_frames"]))] = float(
            entry["confidence"])
    return table


def render(description, touch, confidence, decoder=DECODER, weight=SHAPE_WEIGHT, switch_dead=SWITCH_DEAD,
           min_dead=MIN_DEAD, dead_weight=DEAD_WEIGHT, scope=SCOPE, shape=True, shift=None):
    """The TOML text of one config. `shift` is None (no refiner) or (team, into_dead, out_of_dead) frames."""
    radius, speed, min_frames, hold = touch
    lines = [
        '# P27 (docs/superpowers/specs/2026-11-09-p27-degradation-design.md): written by pipeline/benchmark/p27_grid.py.',
        '',
        'description = "{}"'.format(description),
        'seed = 0',
        'scope = "halves"',
        '',
        '[[evidence]]',
        'kind = "controlled_touch"',
        'radius = {!r}'.format(float(radius)),
        'max_rel_speed = {!r}'.format(float(speed)),
        'min_frames = {}'.format(int(min_frames)),
        'confidence = {!r}    # touch_stats.py --grid r013 --profile generic: floor of the pooled agreement'.format(
            float(confidence)),
        'hold = {!r}'.format(float(hold)),
        '',
        '[[evidence]]',
        'kind = "dead_ball_model"',
        'weight = {!r}'.format(float(dead_weight)),
        'model = "pipeline/models/dead_ball_hgb.json"',
    ]
    if shape:
        lines += [
            '',
            '[[evidence]]',
            'kind = "shape_possession"',
            'weight = {!r}'.format(float(weight)),
            'model = "pipeline/models/shape_possession_hgb.json"',
            'scope = "{}"'.format(scope),
            'min_prob = {!r}'.format(MIN_PROB),
        ]
    lines += [
        '',
        '[decoder]',
        'kind = "hmm"',
        'switch_team = {}'.format(int(decoder[0])),
        'switch_dead = {}'.format(int(switch_dead)),
        'min_possession = {}'.format(int(decoder[1])),
        'min_dead = {}'.format(int(min_dead)),
    ]
    if shift is not None:
        lines += [
            '',
            '[[refine]]',
            'kind = "shift_edges"',
            'team = {}'.format(int(shift[0])),
            'into_dead = {}'.format(int(shift[1])),
            'out_of_dead = {}'.format(int(shift[2])),
        ]
    return "\n".join(lines) + "\n"


def write(stem, text):
    os.makedirs(FOLDER, exist_ok=True)
    with open(os.path.join(FOLDER, stem + ".toml"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return stem


def confidence_of(table, touch):
    key = (float(touch[0]), float(touch[1]), int(touch[2]))
    if key not in table:
        raise SystemExit("p27_grid: no measured confidence for radius {} m, speed {} m/s, min_frames {}".format(*key))
    return table[key]


def touch_stem(touch):
    return "r{:02d}_v{:02d}_k{}_h{:02d}".format(int(round(touch[0] * 10)), int(round(touch[1])), int(touch[2]),
                                                  int(round(touch[3] * 100)))


def reference(table):
    """p27_ref: the incumbent constants with their confidence measured on generic, no shift."""
    return [write("p27_ref", render("P27 reference: main.toml's constants (R024), pre-shift, on the retrained models",
                                    TOUCH, confidence_of(table, TOUCH)))]


def stage_a(table):
    """192 configs; tie order: larger min_frames, smaller radius, smaller speed, smaller hold."""
    settings = [(r, v, k, h) for r in RADII for v in SPEEDS for k in MIN_FRAMES for h in HOLDS]
    settings.sort(key=lambda s: (-s[2], s[0], s[1], s[3]))
    return [write("p27a_" + touch_stem(s), render("P27 Stage A: the touch rule {} m / {} m/s / {} frames / hold {}"
                                                  .format(*s), s, confidence_of(table, s))) for s in settings]


def stage_c(table, touch):
    """144 configs; tie order: larger switch_team, larger min_possession, smaller weight."""
    settings = [(t, mp, w) for t in SWITCH_TEAMS for mp in MIN_POSSESSIONS for w in WEIGHTS]
    settings.sort(key=lambda s: (-s[0], -s[1], s[2]))
    confidence = confidence_of(table, touch)
    return [write("p27c_t{:02d}_mp{:03d}_w{:03d}".format(t, mp, int(round(w * 1000))),
                  render("P27 Stage C: switch_team {} / min_possession {} / shape weight {}".format(t, mp, w), touch,
                         confidence, decoder=(t, mp), weight=w)) for t, mp, w in settings]


def stage_d(table, touch, decoder, weight):
    """Stage D's groups at Stage C's choice; the incumbent of every group is the Stage C config itself (p27d_base)."""
    confidence = confidence_of(table, touch)
    common = dict(touch=touch, confidence=confidence, decoder=decoder, weight=weight)
    stems = [write("p27d_base", render("P27 Stage D: Stage C's choice, the incumbent of every group", **common))]
    for sd in SWITCH_DEADS:
        for md in MIN_DEADS:
            if (sd, md) != (SWITCH_DEAD, MIN_DEAD):
                stems.append(write("p27d_sd{:02d}_md{:02d}".format(sd, md), render(
                    "P27 Stage D: switch_dead {} / min_dead {}".format(sd, md), switch_dead=sd, min_dead=md, **common)))
    for dw in DEAD_WEIGHTS:
        if dw != DEAD_WEIGHT:
            stems.append(write("p27d_dw{:03d}".format(int(round(dw * 100))), render(
                "P27 Stage D: dead-ball weight {}".format(dw), dead_weight=dw, **common)))
    for scope in SCOPES:
        if scope != SCOPE:
            stems.append(write("p27d_scope_{}".format(scope), render(
                "P27 Stage D: shape scope {}".format(scope), scope=scope, **common)))
    stems.append(write("p27d_noshape", render("P27 Stage D: without the shape source", shape=False, **common)))
    return stems


def numbers(text, kinds):
    parts = text.split(",")
    if len(parts) != len(kinds):
        raise SystemExit("p27_grid: expected {} comma-separated values, got {!r}".format(len(kinds), text))
    return tuple(kind(part) for kind, part in zip(kinds, parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the configs of P27's re-choice.")
    parser.add_argument("stage", choices=("ref", "a", "c", "d", "final"))
    parser.add_argument("--confidences", required=True, help="the touch_stats result file (generic, grid r013)")
    parser.add_argument("--touch", help="radius,max_rel_speed,min_frames,hold")
    parser.add_argument("--decoder", help="switch_team,min_possession")
    parser.add_argument("--weight", type=float, help="the shape source's weight")
    parser.add_argument("--switch-dead", type=int, default=SWITCH_DEAD)
    parser.add_argument("--min-dead", type=int, default=MIN_DEAD)
    parser.add_argument("--dead-weight", type=float, default=DEAD_WEIGHT)
    parser.add_argument("--scope", choices=SCOPES, default=SCOPE)
    parser.add_argument("--no-shape", action="store_true")
    parser.add_argument("--shift", help="team,into_dead,out_of_dead (frames); omitted: no refiner")
    parser.add_argument("--name", default="p27_final", help="the stem of the final config")
    args = parser.parse_args(argv)
    table = confidences(args.confidences)
    touch = numbers(args.touch, (float, float, int, float)) if args.touch else None
    decoder = numbers(args.decoder, (int, int)) if args.decoder else None
    if args.stage == "ref":
        stems = reference(table)
    elif args.stage == "a":
        stems = stage_a(table)
    elif args.stage == "c":
        stems = stage_c(table, touch)
    elif args.stage == "d":
        stems = stage_d(table, touch, decoder, args.weight)
    else:
        shift = numbers(args.shift, (int, int, int)) if args.shift else None
        stems = [write(args.name, render(
            "P27: the chosen pipeline{}".format("" if shift else ", pre-shift"), touch, confidence_of(table, touch),
            decoder=decoder, weight=args.weight, switch_dead=args.switch_dead, min_dead=args.min_dead,
            dead_weight=args.dead_weight, scope=args.scope, shape=not args.no_shape, shift=shift))]
    print(" ".join(stems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
