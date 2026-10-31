"""R021's grid: the configs of Stage 1 and Stage 2 written from main.toml's text, their tie order, and the bar's verdict.

    python pipeline/benchmark/team_grid.py stage1 [--main pipeline/configs/main.toml] [--out-dir DIR]
    python pipeline/benchmark/team_grid.py stage2 CHOSEN.toml [--prefix r021b] [--switch-team 4 8 16]
                                                              [--min-possession 35 50 75] [--out-dir DIR]
    python pipeline/benchmark/team_grid.py order CONFIG.toml ...
    python pipeline/benchmark/team_grid.py verdict --stage S.json --union U.json --context C.json
                                                   --union-configs STAGE1.toml ... STAGE2.toml ...

What is declared (docs/results.md R021 part 1) and what this file does with it
-----------------------------------------------------------------------------
Stage 1 is eight configs, `team_model` at weight {0.1, 0.25, 0.5, 1.0} x `shape_possession` scope {all, gaps}, each
`main.toml`'s text with a `team_model` block added and, for `gaps`, the one `scope = "all"` line changed; the decoder
and the shift are `main.toml`'s own. Stage 2 is nine configs of Stage 1's choice, `switch_team` x `min_possession`,
two decoder lines changed. Nothing else in a config may differ, so the files are made as `restarts.write_configs` makes
R014's: the text is copied line by line and an anchor that is not found exactly once is an error, never a guess. Every
file is written with LF endings by `headroom.write_text`, so that the same input writes the same bytes twice, and a
config of R014's kind (kept verbatim, one block and a description added) is what `load_config` reads back.

The stems are the file names. Their order is the declared tie order that `choose.py` is given: by weight, then `all`
before `gaps` (Stage 1); by `switch_team`, then `min_possession` (Stage 2). `union_order` gives the order of the bar's
leave-one-match-out (R021's C2): the configs of both stages sorted stably by (team_model weight, scope with `all` first,
`switch_team`, `min_possession`), each distinct config once, a config whose evidence, decoder and refiners equal an
earlier one's being dropped (Stage 2's `r021b_t04_mp035` is Stage 1's choice again, so Stage 1's paths go first and its
stem is the one kept). A Stage 2b of R021c's stems goes through the same function with the stems of the earlier
stages first.

The verdict (`verdict`) is the declared bar applied to the numbers that `choose.py --out` wrote, with no rounding and no
judgement: a miss by any margin is a miss (advisor-01.md section 3.2, A14's -0.0003).

    bar      the chosen config's raw pooled delta over main.toml is >= +0.005 and its sign-flip p is <= 0.05
    veto     fires if any of the coarse delta, the `context` raw delta, the `context` coarse delta, or the alive/dead F1
             delta (the raw report's rows) is < -0.005
    lomo     the union's whole-rule leave-one-match-out raw mean delta is > 0 with at least 6 wins in 7
    passes   the bar, no veto and the LOMO
    retry_2b the raw bar and the LOMO pass and the only vetoes that fire are the coarse ones or `context`'s (R021's
             pre-approved Stage 2b)
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

from pipeline.benchmark import run  # noqa: E402
from pipeline.benchmark.headroom import write_text  # noqa: E402
from pipeline.config import load_config  # noqa: E402

# Params
DEFAULT_MAIN = os.path.join(ROOT, "pipeline", "configs", "main.toml")
DEFAULT_FOLDER = os.path.join(ROOT, "pipeline", "benchmark", "configs", "r021")
MODEL = "pipeline/models/team_model_hgb.json"
MIN_PROB = 0.02
WEIGHTS = (0.1, 0.25, 0.5, 1.0)
SCOPES = ("all", "gaps")
SWITCH_TEAM = (4, 8, 16)
MIN_POSSESSION = (35, 50, 75)
BAR_DELTA = 0.005                   # R014's: the chosen config's raw pooled delta over main.toml
BAR_P = 0.05                        # and its exact sign-flip p
VETO = -0.005                       # a delta below this vetoes
LOMO_WINS = 6                       # of 7 matches
REFERENCE = "main"                  # what every document is a comparison with
PROFILE, CONTEXT = "tracking", "context"


# -- the configs ------------------------------------------------------------------------------------------------------

def team_block(weight, eol="\n"):
    """The `[[evidence]]` entry of the team model at `weight`, ending with a blank line, as one string."""
    lines = ["[[evidence]]", 'kind = "team_model"', "weight = {}".format(_number(weight)),
             'model = "{}"   # relative to the repo root; BAR-ATH has no fold, so `full`'.format(MODEL),
             "min_prob = {}".format(_number(MIN_PROB)), ""]
    return eol.join(lines) + eol


def stage1(main_path=DEFAULT_MAIN, folder=DEFAULT_FOLDER, weights=WEIGHTS, scopes=SCOPES):
    """The paths of Stage 1's configs, `r021a_w{WWWW}_{all|gaps}.toml` in the declared tie order, written to `folder`.

    Each is main.toml's text with a `team_model` block before `[decoder]` (and the comment lines that head it), the one
    `scope = "all"` line changed to "gaps" for the `gaps` configs, and the description amended. ValueError for an anchor
    found other than once.
    """
    lines = _read(main_path)
    decoder = _only(lines, lambda line: line.strip() == "[decoder]", main_path, "a [decoder] line")
    description = _only(lines, lambda line: line.startswith("description = "), main_path, "a 'description = ' line")
    scope = _only(lines, lambda line: line.startswith('scope = "all"'), main_path, 'a \'scope = "all"\' line')
    _only(lines, lambda line: line.strip() == 'kind = "shape_possession"', main_path, "a shape_possession entry")
    body = lines[description]
    if not body.endswith('"'):
        raise ValueError("team_grid: the description line of {} does not end with a quote".format(main_path))
    at = decoder
    while at > 0 and lines[at - 1].startswith("#"):
        at -= 1
    for name in scopes:
        if name not in SCOPES:
            raise ValueError("team_grid: scope must be one of {}, got {!r}".format(SCOPES, name))
    paths = []
    for weight in weights:
        for name in scopes:
            amended = list(lines)
            amended[description] = body[:-1] + ' + team_model (P19, R021) at weight {}; shape_possession scope {}"'.format(
                _number(weight), name)
            if name == "gaps":
                amended[scope] = lines[scope].replace('scope = "all"', 'scope = "gaps"', 1)
            amended[at:at] = team_block(weight).splitlines()
            stem = "r021a_w{:04d}_{}".format(int(round(weight * 1000)), name)
            path = os.path.join(folder, stem + ".toml")
            write_text(path, "\n".join(amended) + "\n")
            paths.append(path)
    return paths


def stage2(chosen_path, folder=DEFAULT_FOLDER, prefix="r021b", switch_team=SWITCH_TEAM,
           min_possession=MIN_POSSESSION):
    """The paths of Stage 2's configs, `{prefix}_t{TT}_mp{PPP}.toml` by `switch_team` then `min_possession`.

    Each is the chosen config's text with the one `switch_team =` and the one `min_possession =` line of `[decoder]`
    given the grid's values (a comment after the value stays). ValueError for an anchor found other than once.
    """
    lines = _read(chosen_path)
    team = _only(lines, lambda line: re.match(r"switch_team\s*=", line), chosen_path, "a 'switch_team =' line")
    possession = _only(lines, lambda line: re.match(r"min_possession\s*=", line), chosen_path,
                       "a 'min_possession =' line")
    paths = []
    for switch in switch_team:
        for minimum in min_possession:
            amended = list(lines)
            amended[team] = re.sub(r"^(switch_team\s*=\s*)\S+", lambda m: m.group(1) + str(switch), lines[team])
            amended[possession] = re.sub(r"^(min_possession\s*=\s*)\S+", lambda m: m.group(1) + str(minimum),
                                         lines[possession])
            path = os.path.join(folder, "{}_t{:02d}_mp{:03d}.toml".format(prefix, switch, minimum))
            write_text(path, "\n".join(amended) + "\n")
            paths.append(path)
    return paths


def stems(paths):
    """The stems (file names without .toml) of `paths`, in the order given: the declared tie order."""
    return [os.path.splitext(os.path.basename(path))[0] for path in paths]


def union_order(paths):
    """The stems of `paths` in the order of R021's union leave-one-match-out: sorted, each distinct config once.

    Stable sort by (team_model weight, shape_possession scope with `all` first, switch_team, min_possession); then a
    config whose evidence, decoder and refiners equal an earlier one's is dropped. Pass the earlier stage's paths first:
    among equal keys the stable sort keeps their order, so the earlier stage's stem is the one that survives. ValueError
    for a config that has no team_model or no shape_possession entry.
    """
    keyed = []
    for path in paths:
        config = load_config(path)
        team = [part for part in config.evidence if part.kind == "team_model"]
        shape = [part for part in config.evidence if part.kind == "shape_possession"]
        if len(team) != 1 or len(shape) != 1:
            raise ValueError("team_grid: {} must have one team_model and one shape_possession entry, it has {} and {}"
                             .format(path, len(team), len(shape)))
        scope = shape[0].params.get("scope")
        if scope not in SCOPES:
            raise ValueError("team_grid: {} must give its shape_possession entry a scope ({}), got {!r}".format(
                path, " or ".join(SCOPES), scope))
        key = (team[0].weight, SCOPES.index(scope), config.decoder.params["switch_team"],
               config.decoder.params["min_possession"])
        keyed.append((key, config))
    keyed.sort(key=lambda item: item[0])                 # stable: equal keys keep the order the paths came in
    kept = []
    for _, config in keyed:
        if not any(_same(config, other) for other in kept):
            kept.append(config)
    return [config.stem for config in kept]


def _same(a, b):
    return a.evidence == b.evidence and a.decoder == b.decoder and a.refine == b.refine


# -- the verdict ------------------------------------------------------------------------------------------------------

def verdict(stage_choice, union_choice, context_choice, union_stems):
    """The declared bar on the `choose.py --out` documents, mechanically. See the module docstring.

    `stage_choice` is the choice of the stage whose chosen config is tested (reference main, profile tracking),
    `union_choice` the choice over the union of the stages' distinct configs (its `lomo`), `context_choice` a
    `--profile context` run of main and the chosen config alone, and `union_stems` the declared union (`union_order` of
    the stages' paths). ValueError, loudly, unless each document is the one it should be: the reference is main, the
    profile is tracking for the first two and context for the third, the metric is the spellF1, the chosen config is the
    one the stage chose and is in the union, the context run holds that one config, and the union's stems are
    `union_stems` exactly, in that order. A verdict on the wrong file would be a number that means nothing.
    Returns {chosen, raw, vetoes, lomo, passes, retry_2b}.
    """
    _check_documents(stage_choice, union_choice, context_choice, list(union_stems))
    chosen, reference = stage_choice["chosen"], stage_choice["reference"]
    raw = stage_choice["versus"]["raw"]
    bar = {"delta": raw["pooled_delta"], "p": raw["p_sign_flip"]}
    bar["passes"] = bool(bar["delta"] >= BAR_DELTA and bar["p"] <= BAR_P)
    rows = stage_choice["rows"]
    deltas = {"coarse": stage_choice["versus"]["coarse"]["pooled_delta"],
              "context_raw": context_choice["versus"]["raw"]["pooled_delta"],
              "context_coarse": context_choice["versus"]["coarse"]["pooled_delta"],
              # the reports carry the F1s to 9 decimals: the difference is rounded there, so that -0.005 is -0.005
              "alive_dead": round(rows[chosen]["alive_dead"] - rows[reference]["alive_dead"], 9)}
    vetoes = {name: {"delta": delta, "fires": bool(delta < VETO)} for name, delta in deltas.items()}
    entry = union_choice["lomo"]["rule"]["raw"]
    lomo = {"mean": entry["mean"], "wins": entry["wins"], "passes": bool(entry["mean"] > 0 and entry["wins"] >= LOMO_WINS)}
    fired = [name for name, cell in vetoes.items() if cell["fires"]]
    passes = bar["passes"] and not fired and lomo["passes"]
    retry = bool(bar["passes"] and lomo["passes"] and fired and "alive_dead" not in fired)
    return {"chosen": chosen, "raw": bar, "vetoes": vetoes, "lomo": lomo, "passes": passes, "retry_2b": retry}


def _check_documents(stage, union, context, union_stems):
    def refuse(sentence):
        raise ValueError("team_grid: verdict refused: {}".format(sentence))

    for name, document, profile in (("stage", stage, PROFILE), ("union", union, PROFILE), ("context", context, CONTEXT)):
        for key in ("reference", "profile", "metric", "chosen", "order"):
            if key not in document:
                refuse("the {} file has no {!r}; it is not a choose.py --out file".format(name, key))
        if document["reference"] != REFERENCE:
            refuse("the {} file's reference is {!r}, not {!r}".format(name, document["reference"], REFERENCE))
        # choose.py writes the profile as the runner's record, {"name", "version", ...}, not as a bare name
        found = document["profile"].get("name") if isinstance(document["profile"], dict) else None
        if found != profile:
            refuse("the {} file is of profile {!r}, it must be {!r}".format(name, found, profile))
        if document["metric"] != "spell":
            refuse("the {} file's metric is {!r}, it must be 'spell'".format(name, document["metric"]))
    chosen = stage["chosen"]
    if context["chosen"] != chosen or list(context["order"]) != [chosen]:
        refuse("the context file is of {} (chosen {!r}), not of the stage's chosen config {!r} alone".format(
            list(context["order"]), context["chosen"], chosen))
    if chosen not in union_stems:
        refuse("the stage's chosen config {!r} is not in the declared union".format(chosen))
    if list(union["order"]) != union_stems:
        refuse("the union file's configs {} are not the declared union {}, in order".format(
            list(union["order"]), union_stems))


def render_verdict(result):
    lines = ["verdict   chosen {}".format(result["chosen"]),
             "verdict   raw delta {:+.4f} (>= +{}), sign-flip p {:.4f} (<= {}): {}".format(
                 result["raw"]["delta"], BAR_DELTA, result["raw"]["p"], BAR_P,
                 "met" if result["raw"]["passes"] else "NOT met")]
    for name, cell in result["vetoes"].items():
        lines.append("verdict   veto {:<15} {:+.4f} (fires below {}): {}".format(
            name, cell["delta"], VETO, "FIRES" if cell["fires"] else "no"))
    lines.append("verdict   union LOMO raw mean {:+.4f}, {} wins (> 0 and >= {}): {}".format(
        result["lomo"]["mean"], result["lomo"]["wins"], LOMO_WINS, "met" if result["lomo"]["passes"] else "NOT met"))
    lines.append("verdict   {}".format("PASSES: adopt as R021's adoption rule says" if result["passes"]
                                      else "does not pass"))
    lines.append("verdict   Stage 2b (the one retry) is {}".format("pre-approved" if result["retry_2b"]
                                                                    else "not pre-approved"))
    return lines


# -- small things -----------------------------------------------------------------------------------------------------

def _read(path):
    """The lines of the file at `path` with LF endings and no line end kept."""
    with open(path, newline="", encoding="utf-8") as handle:
        lines = handle.read().replace("\r\n", "\n").split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _only(lines, test, path, what):
    """The index of the one line that satisfies `test`; ValueError if there is none or more than one."""
    found = [index for index, line in enumerate(lines) if test(line)]
    if len(found) != 1:
        raise ValueError("team_grid: {} has {} of {}, want exactly 1".format(path, len(found), what))
    return found[0]


def _number(value):
    """A TOML float: 0.1, 0.25, 0.5, 1.0 as Python writes them."""
    return repr(float(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description="R021's configs, their order and the bar's verdict.")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("stage1", help="write Stage 1's 8 configs from main.toml")
    one.add_argument("--main", default=DEFAULT_MAIN)
    one.add_argument("--out-dir", default=DEFAULT_FOLDER)
    two = commands.add_parser("stage2", help="write Stage 2's configs from Stage 1's choice")
    two.add_argument("chosen")
    two.add_argument("--prefix", default="r021b")
    two.add_argument("--switch-team", type=int, nargs="+", default=list(SWITCH_TEAM))
    two.add_argument("--min-possession", type=int, nargs="+", default=list(MIN_POSSESSION))
    two.add_argument("--out-dir", default=DEFAULT_FOLDER)
    order = commands.add_parser("order", help="the union's stems, in order, each distinct config once")
    order.add_argument("configs", nargs="+")
    final = commands.add_parser("verdict", help="the bar on three choose.py --out files")
    final.add_argument("--stage", required=True)
    final.add_argument("--union", required=True)
    final.add_argument("--context", required=True)
    final.add_argument("--union-configs", nargs="+", required=True, metavar="CONFIG",
                       help="the stages' configs, Stage 1's first: the declared union that --union must be of")
    args = parser.parse_args(argv)
    try:
        if args.command == "stage1":
            print("\n".join(stems(stage1(args.main, args.out_dir))))
        elif args.command == "stage2":
            print("\n".join(stems(stage2(args.chosen, args.out_dir, args.prefix, args.switch_team,
                                         args.min_possession))))
        elif args.command == "order":
            print(" ".join(union_order(args.configs)))
        else:
            import json
            documents = []
            for path in (args.stage, args.union, args.context):
                with open(path, encoding="utf-8") as handle:
                    documents.append(json.load(handle))
            result = verdict(*documents, union_order(args.union_configs))
            for line in render_verdict(result):
                print(line)
        return 0
    except (ValueError, OSError, run.BenchmarkError) as error:
        print("FAILED    {}".format(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
