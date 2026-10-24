"""Headroom: what a perfect dead-ball detector, or a perfect possessor, would be worth to a config.

    python pipeline/benchmark/headroom.py CONFIG [--profile P] [--seeds N] [--truth raw|coarse] [--matches ID ...]
                                          [--data-dir D] [--write]

    result = run_headroom(config_path, profile=..., truth=..., write=False)   # evaluate_configs' result, for 6 configs

A decoder can only be as good as the evidence it is given, and the evidence we have is weak where it matters: the ball
is missing on 41% of graded frames, and with it visible the nearest player's team is wrong 22% of the time (A01, A02).
Before building a dead-ball detector (P05) or a possession model (P04) it is worth knowing how much of the score each
could ever deliver, to a decoder that is already there. This runs CONFIG on the benchmark beside five variants of it
whose evidence is built from the benchmark's own truth, perfectly, and the runner's table says by how much each beats
it:

    CONFIG                      the reference, run from its own file as it is
    <stem>__alive               CONFIG's evidence plus oracle_alive: a perfect dead-ball detector
    <stem>__home_visible        nearest_player replaced by oracle_home_visible: the true possessor on the frames where
                                the tracking has the ball, nothing where it does not
    <stem>__home_all            nearest_player replaced by oracle_home_all: the true possessor on every live frame, the
                                ball seen or not (what a model that does not need the ball could reach)
    <stem>__alive_home_visible  both of the above
    <stem>__alive_home_all      both, whole: with a decoder free to follow them it reproduces the truth

The oracles
-----------
* oracle_alive: alive is 1 where the truth is H or A and 0 where it is D; home is NaN. NaN on X, outside the halves.
* oracle_home_visible: home is 1 where the truth is H and 0 where it is A, on the frames where the tracking has the ball
  (run.inputs.atd.ball_present, which a degraded profile has made as it is in our match); NaN elsewhere. alive is NaN.
* oracle_home_all: the same on every H and A frame. alive is NaN.

They are ordinary evidence sources of exactly 1.0 and 0.0 and of weight 1, so a probabilistic decoder reads them as
about 13.8 nats a frame at eps 1e-6: a bound on what the evidence can give, not an estimate of what a real source will.
A replaced nearest_player keeps its weight and gives its place to the oracle (named as its kind, without its
`confidence`); oracle_alive is added after the config's own sources at weight 1, and a ball_presence the config has
stays, though next to the oracle it counts for nothing. Whatever a decoder, its penalties and the refiners do with a
perfect source is the measurement: the headroom is that of this config, not of the benchmark.

Where the truth comes from, and where the oracles live
------------------------------------------------------
Nothing of the truth reaches the inputs: a MatchInputs has none, and the leak checks (checks/common.leaks) look for it.
The oracles read it by closure, from a dict keyed on the match id, that is filled as the runner loads each match: the
truth is truth.truth_codes of the match that evaluate_configs itself was handed (coarse_codes of it under truth =
"coarse"), and an oracle finds it by run.inputs.game. Without `matches`, the runner is handed a Mapping that loads the
seven DFL matches when asked for, one at a time, as it loads them itself (LoadedMatches), and keeps nothing, so that a
headroom run needs the same memory as any other.

An oracle is a source that cannot be allowed to reach a submission, and cannot: it exists only inside `registered`, a
context manager that puts the three sources into pipeline.evidence.REGISTRY for the length of the run and restores the
registry afterwards, even if the run raises (the pattern of checks/run.py, written again here because production code
does not import checks). Importing this module registers nothing (test_headroom_never_in_production), and a config
that names an oracle does not load outside a headroom run. pipeline/decode/ and pipeline/evidence/ know nothing of it.

The variants are configs
------------------------
They are written as TOML text into <data dir>/runs/benchmark/headroom/configs/<stem>__<variant>.toml, a fixed folder,
so that a config's path, its text (both are in its report) and so its report's bytes are the same on every run. They are
rewritten each time, and read by the pipeline's own loader like any config: the benchmark is only worth anything if it
scores what the submission pipeline does. The writer handles the flat tables that a Config holds, str, int, float and
bool, and refuses a parameter of another kind by name (toml_value) before any file is written.

Reports are written only with --write (write=True), as the runner writes them: <stem>[__<variant>].json under
<data dir>/runs/benchmark/<profile>/, with the same suffixes for a subset of matches, the coarse truth and a number of
seeds. The CONFIG's own report is the one the runner would write for it, so a headroom run with --write replaces a
report of CONFIG made earlier on the same matches, with the same numbers.

What it does not tell
---------------------
The headroom is an upper bound, and a loose one: no real detector is perfect, and a perfect one that also reads the
truth at the frames the ball is missing is not one the tracking could build. It says where not to spend time (a small
gap is a small prize) and in which order to think about the work, and it is read like every comparison of the
benchmark, across the seven matches and under both truths (README, "Headroom"). Nothing here is a config to submit.
"""

import argparse
import collections.abc
import contextlib
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here (run.py, inputs.py, ...) can shadow a
    # module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline import evidence as evidence_stage  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import dfl, run, sources  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import Component, ConfigError, load_config  # noqa: E402
from pipeline.contracts import A, D, H, Evidence, no_opinion  # noqa: E402

# Params
DEFAULT_PROFILE = run.DEFAULT_PROFILE
FOLDER = ("runs", "benchmark", "headroom", "configs")   # under the data dir: where the variants are written
# The kinds of source that say who has the ball, one of which the home variants replace.
HOME_KINDS = ("nearest_player",)
ALIVE_ORACLE = "oracle_alive"
# (suffix of the stem, alive oracle added, home oracle replacing the source, what it measures), in the order they run.
VARIANTS = (
    ("alive", True, None,
     "a perfect dead-ball detector: oracle_alive added, who has the ball from the config's own evidence"),
    ("home_visible", False, "oracle_home_visible",
     "a perfect possessor where the ball is seen: the home source replaced, silent where the ball is missing"),
    ("home_all", False, "oracle_home_all",
     "a perfect possessor on every live frame, the ball seen or not: the home source replaced"),
    ("alive_home_visible", True, "oracle_home_visible",
     "both: a perfect dead-ball detector and a perfect possessor where the ball is seen"),
    ("alive_home_all", True, "oracle_home_all",
     "both, whole: the truth itself, which a decoder free to follow it reproduces"),
)
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+\Z")               # a TOML key that needs no quotes
ESCAPES = {"\n": "\\n", "\t": "\\t", "\r": "\\r", "\b": "\\b", "\f": "\\f", '"': '\\"', "\\": "\\\\"}


# -- the oracles ------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def registered(registry, **components):
    """The components in `registry` (a stage's REGISTRY) while the block runs, and the registry as it was after it.

    The registry is restored whatever happens inside, an exception included, by clearing it and putting back what it
    held before: what was added is gone and what was replaced is back.
    """
    before = dict(registry)
    registry.update(components)
    try:
        yield
    finally:
        registry.clear()
        registry.update(before)


def answers(match, truth):
    """The truth's int8 codes for `match`, as the runner scores against them: truth_codes, coarse_codes of them for
    "coarse". Read-only."""
    if truth not in run.TRUTHS:
        raise run.BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(run.TRUTHS)))
    codes, _ = truths.truth_codes(match)
    if truth == "coarse":
        codes = truths.coarse_codes(codes, match.graded, match.fps)
    codes.setflags(write=False)
    return codes


def oracles(codes_of):
    """{kind: compute} for the three oracle sources, reading the truth from `codes_of`.

    `codes_of` maps a match id to its truth codes (answers); it is read when a source is computed, by run.inputs.game,
    so it may be filled while the run goes (LoadedMatches does). A source for a match that is not in it raises
    ValueError saying so: an oracle never guesses.
    """
    def truth_of(kind, run_context):
        game, n_frames = run_context.inputs.game, run_context.inputs.n_frames
        if game not in codes_of:
            raise ValueError("{}: no truth was made for match {!r}; headroom.run_headroom makes it of each match that "
                             "it runs, so call the source through it".format(kind, game))
        codes = codes_of[game]
        if len(codes) != n_frames:
            raise ValueError("{}: the truth of match {!r} has {} frames and the inputs {}".format(
                kind, game, len(codes), n_frames))
        return codes

    def oracle_alive(run_context):
        codes = truth_of(ALIVE_ORACLE, run_context)
        alive = no_opinion(len(codes))
        alive[(codes == H) | (codes == A)] = 1.0
        alive[codes == D] = 0.0
        return Evidence(alive=alive, home=no_opinion(len(codes)))

    def home_oracle(kind, visible):
        def compute(run_context):
            codes = truth_of(kind, run_context)
            speaks = (codes == H) | (codes == A)
            if visible:
                speaks &= run_context.inputs.atd.ball_present
            home = no_opinion(len(codes))
            home[speaks] = np.where(codes[speaks] == H, 1.0, 0.0)
            return Evidence(alive=no_opinion(len(codes)), home=home)
        return compute

    return {ALIVE_ORACLE: oracle_alive,
            "oracle_home_visible": home_oracle("oracle_home_visible", True),
            "oracle_home_all": home_oracle("oracle_home_all", False)}


class LoadedMatches(collections.abc.Mapping):
    """The registry's seven matches as the runner reads them, with the truth of each made as it is loaded.

    evaluate_configs takes `matches` as {id: DflMatch} and reads one with matches[id] when its turn comes, once, and
    lets go of it before the next. This is that dict without the seven in memory: a lookup loads the match by
    dfl.load_match, from `data_dir` (default games.DATA_DIR), puts its truth (answers) into `codes_of` and returns the
    match, keeping neither the match nor a reference to it. The ids are sources.MATCH_IDS in their order; asking whether
    an id is there does not load it.
    """

    def __init__(self, codes_of, truth, data_dir=None):
        self._codes_of = codes_of
        self._truth = truth
        self._data_dir = data_dir

    def __getitem__(self, match_id):
        if match_id not in sources.MATCH_IDS:
            raise KeyError(match_id)
        match = dfl.load_match(match_id, data_dir=self._data_dir)
        self._codes_of[match_id] = answers(match, self._truth)
        return match

    def __contains__(self, match_id):
        return match_id in sources.MATCH_IDS

    def __iter__(self):
        return iter(sources.MATCH_IDS)

    def __len__(self):
        return len(sources.MATCH_IDS)


# -- the variants -----------------------------------------------------------------------------------------------------

def check_config(config):
    """BenchmarkError unless `config` can be run by the runner and has the one home source the variants replace.

    The runner's own refusals come first, in its words and before anything is written (a per-file decode, a dot in the
    name, which the report's file name could not carry), then the home source: none, or more than one, leaves the home
    variants nothing to replace, or too much.
    """
    run.check_scope(config)
    if "." in config.stem:
        raise run.BenchmarkError(
            "{}: the file name of a config is the name of its report, and the variants' names are made from it; a dot "
            "in it would be read as one of the report's parts (<name>.coarse.m-<ids>.s<n>.json), so rename {} without "
            "a dot".format(config.stem, run.shown(config.path)))
    home_source(config)


def home_source(config):
    """The evidence Component that the home oracles replace; BenchmarkError unless `config` has exactly one."""
    found = [component for component in config.evidence if component.kind in HOME_KINDS]
    if not found:
        raise run.BenchmarkError(
            "{}: no evidence source of kind {} to replace with the home oracles (the variants home_visible, home_all, "
            "alive_home_visible and alive_home_all take its place); add one, or run a config that has it".format(
                config.stem, " or ".join(HOME_KINDS)))
    if len(found) > 1:
        raise run.BenchmarkError(
            "{}: {} evidence sources of kind {} ({}); the home oracles replace one, and which one is not for the "
            "runner to say; keep one".format(config.stem, len(found), " or ".join(HOME_KINDS),
                                             ", ".join(repr(component.name) for component in found)))
    return found[0]


def variant_evidence(config, alive, home):
    """The evidence of a variant: `config`'s own, the home source replaced by the oracle `home` (None: it stays) in its
    place with its weight, and ALIVE_ORACLE added after the rest if `alive`."""
    entries = list(config.evidence)
    if home is not None:
        replaced = home_source(config)
        at = entries.index(replaced)
        entries[at] = Component(kind=home, name=home, weight=replaced.weight, params={})
    if alive:
        entries.append(Component(kind=ALIVE_ORACLE, name=ALIVE_ORACLE, weight=1.0, params={}))
    return tuple(entries)


def variant_texts(config):
    """[(stem, TOML text)] of the five variants of `config`, in the order they run. BenchmarkError if any cannot be made
    (no home source, a parameter that cannot be written), so that nothing has been written when it is raised."""
    source = run.shown(config.path)
    texts = []
    for suffix, alive, home, meaning in VARIANTS:
        stem = "{}__{}".format(config.stem, suffix)
        note = ["Written by pipeline/benchmark/headroom.py from {}, and rewritten by every run: edit that, not "
                "this.".format(source),
                "Variant {}: {}.".format(suffix, meaning),
                "Its oracle sources read the benchmark's truth and exist only while headroom.py runs. Not a config to "
                "submit."]
        description = "{}Headroom variant of {}: {}.".format(config.description + " " if config.description else "",
                                                             config.stem, meaning)
        texts.append((stem, toml_text(config, variant_evidence(config, alive, home), description, note)))
    return texts


def write_variants(config, data_dir=None):
    """Write the variants of `config` into <data dir>/runs/benchmark/headroom/configs/ (data dir default
    games.DATA_DIR); [(stem, path)] in the order they run. Every text is made before the first file is written."""
    folder = os.path.join(data_dir or games.DATA_DIR, *FOLDER)
    texts = variant_texts(config)
    written = []
    for stem, text in texts:
        path = os.path.join(folder, stem + ".toml")
        write_text(path, text)
        written.append((stem, path))
    return written


def write_text(path, text):
    """`text` at `path`, UTF-8 with LF endings, atomically: through a temporary file in the folder, renamed over the
    old one, so that a run that is interrupted never leaves half a config for the next to load."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


# -- the TOML a variant is written as ---------------------------------------------------------------------------------

def toml_text(config, evidence, description, note):
    """The TOML of `config` with `evidence` and `description` in place of its own: `note` lines as comments on top.

    The flat tables that a Config holds: seed and scope, one [[evidence]] per source (its name and weight written only
    where they are not the defaults, the kind and 1.0), the [decoder] and one [[refine]] per refiner, with their
    parameters in the order the config had them. load_config of the result is the config, evidence and description
    aside (test_headroom_variants_round_trip_what_config_holds). BenchmarkError for a parameter that is not a str, an
    int, a float or a bool.
    """
    lines = ["# " + line for line in note] + ["",
                                              "description = " + toml_string(description),
                                              "seed = " + toml_value(config.seed, "seed"),
                                              "scope = " + toml_string(config.scope), ""]
    for component in evidence:
        lines.append("[[evidence]]")
        lines.append("kind = " + toml_string(component.kind))
        if component.name != component.kind:
            lines.append("name = " + toml_string(component.name))
        if component.weight != 1.0:
            lines.append("weight = " + toml_value(component.weight, "weight"))
        lines += _parameters(component, "[[evidence]] {!r}".format(component.kind)) + [""]
    lines += ["[decoder]", "kind = " + toml_string(config.decoder.kind)]
    lines += _parameters(config.decoder, "[decoder] {!r}".format(config.decoder.kind)) + [""]
    for component in config.refine:
        lines += ["[[refine]]", "kind = " + toml_string(component.kind)]
        lines += _parameters(component, "[[refine]] {!r}".format(component.kind)) + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _parameters(component, where):
    return ["{} = {}".format(toml_key(key), toml_value(value, "{} parameter {!r}".format(where, key)))
            for key, value in component.params.items()]


def toml_key(key):
    """`key` as a TOML key: bare where it may be, else quoted."""
    return key if BARE_KEY.match(key) else toml_string(key)


def toml_string(text):
    """`text` as a TOML basic string: quoted, with the quote, the backslash and every control character escaped."""
    pieces = []
    for character in text:
        if character in ESCAPES:
            pieces.append(ESCAPES[character])
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            pieces.append("\\u{:04x}".format(ord(character)))
        else:
            pieces.append(character)
    return '"' + "".join(pieces) + '"'


def toml_value(value, where):
    """`value` as TOML: a bool as true or false, an int as digits, a float as Python writes it (1e-06, inf, nan are
    all TOML's), a str quoted. BenchmarkError naming `where` for anything else, a list or a table included."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return toml_string(value)
    raise run.BenchmarkError(
        "headroom: {} is a {} ({!r}); the variants are written with str, int, float and bool parameters, which is what "
        "the configs of this repository hold, so extend toml_value in pipeline/benchmark/headroom.py before running "
        "this config".format(where, type(value).__name__, value))


# -- the run ----------------------------------------------------------------------------------------------------------

def run_headroom(config_path, *, profile=DEFAULT_PROFILE, seeds=None, truth="raw", match_ids=None, matches=None,
                 calibration=None, data_dir=None, log=print, write=False):
    """Run CONFIG and its five variants through evaluate_configs; its result (CONFIG is the reference).

    The arguments are evaluate_configs' own, with `write`: True writes each config's report as run.write_reports does
    and logs where. The variants are written first into <data dir>/runs/benchmark/headroom/configs/ (write_variants),
    always, since the runner reads them as configs; `log` receives a line for each, then the runner's progress and
    table.
    `matches` is {id: DflMatch} to use instead of loading (the checks pass synthetic ones); its truth is made at once.
    BenchmarkError for what the runner refuses and for a config the variants cannot be made of (check_config), before
    any file is written; ConfigError for one that does not load.
    """
    if truth not in run.TRUTHS:
        raise run.BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(run.TRUTHS)))
    config = load_config(config_path)
    check_config(config)
    written = write_variants(config, data_dir)

    codes_of = {}
    if matches is None:
        served = LoadedMatches(codes_of, truth, data_dir)
    else:
        for match_id in run.resolve_match_ids(match_ids, matches):
            codes_of[match_id] = answers(matches[match_id], truth)
        served = matches

    log("headroom  {}: the variants carry the truth ({}) as evidence, to measure what a perfect source would be worth "
        "to this config; they are not configs to submit".format(config.stem, truth))
    width = max(len(stem) for stem, _ in written)
    for (stem, _), (_, _, _, meaning) in zip(written, VARIANTS):
        log("variant   {:<{}}  {}".format(stem, width, meaning))
    with registered(evidence_stage.REGISTRY, **oracles(codes_of)):
        result = run.evaluate_configs([config.path] + [path for _, path in written], profile=profile, seeds=seeds,
                                      match_ids=match_ids, matches=served, calibration=calibration, truth=truth,
                                      data_dir=data_dir, log=log)
    if write:
        log("")
        for path in run.write_reports(result):
            log("report    {}".format(run.shown(path)))
    return result


# -- the command line -------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run CONFIG on the benchmark beside variants of it whose evidence is the truth (an oracle "
                    "dead-ball detector, an oracle possessor, both): what a perfect source would be worth to it. "
                    "CONFIG is the reference of the comparisons, and has to have one nearest_player source.")
    parser.add_argument("config", metavar="CONFIG",
                        help="a TOML config with one nearest_player source, e.g. "
                             "pipeline/benchmark/configs/nearest_w50.toml")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--truth", choices=run.TRUTHS, default="raw",
                        help="the truth to score against, and that the oracles carry: raw, or coarse")
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--write", action="store_true", help="also write each config's report (default: print only)")
    args = parser.parse_args(argv)
    try:
        run_headroom(args.config, profile=args.profile, seeds=args.seeds, truth=args.truth, match_ids=args.matches,
                     data_dir=args.data_dir, write=args.write)
    except (ConfigError, run.BenchmarkError, OSError) as error:
        # The expected failures get a sentence (as in run.py): a config to fix, a profile or match that does not exist,
        # data not downloaded yet. Anything else is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1
    if not args.write:
        print("")
        print("report    not written (pass --write to write them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
