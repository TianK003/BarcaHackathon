"""Headroom: what a perfect dead-ball detector, or a perfect possessor, would be worth to a config.

    python pipeline/benchmark/headroom.py CONFIG [--profile P] [--seeds N] [--truth raw|coarse] [--matches ID ...]
                                          [--data-dir D] [--drop-match-only] [--write]

    result = run_headroom(config_path, profile=..., truth=..., write=False)   # evaluate_configs' result, for 6 configs

A decoder can only be as good as the evidence it is given, and the evidence we have is weak where it matters: the ball
is missing on 41% of graded frames, and with it visible the nearest player's team is wrong 22% of the time (A01, A02).
Before building a dead-ball detector (P05) or a possession model (P04) it is worth knowing how much of the score each
could ever deliver, to a decoder that is already there. This runs CONFIG on the benchmark beside five variants of it
(and, by name, more: see "The variants are configs") whose evidence is built from the benchmark's own truth, perfectly, and the runner's table says by how much each beats
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
* oracle_home_gaps: the same on the H and A frames where the tracking has no ball (~run.inputs.atd.ball_present), NaN
  where it has. alive is NaN. With oracle_home_visible it makes oracle_home_all; alone it is what a perfect source would
  add where the ball cannot say anything.
* oracle_restart (P17): home is the team that restarts after each stoppage, for the RESTART_TAIL (50) frames from the
  first live frame after a D run, cut short where the truth next changes team; 1 for H, 0 for A. A D run at a block's
  start (the kick-off) counts; one followed by X, or running to the block's end, restarts nothing. NaN elsewhere, and
  alive is NaN (restart_home).
* oracle_restart_run: the same, held until the possession ends (the next frame whose code differs) instead of 50 frames.

They are ordinary evidence sources of exactly 1.0 and 0.0, so a probabilistic decoder reads them as about 13.8 nats a
frame at eps 1e-6 and weight 1: a bound on what the evidence can give, not an estimate of what a real source will. The
alive oracle has weight 1; a home oracle takes the weight of the source it replaces. A replaced nearest_player gives its
place to the oracle (named as its kind, without its `confidence`); oracle_alive is added after the config's own
sources, and a ball_presence the config has stays, though next to the oracle it counts for nothing. Whatever a
decoder, its penalties and the refiners do with a perfect source is the measurement: the headroom is that of this
config, not of the benchmark.

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
so that a config's path, its text (both are in its report) and so its report's bytes are the same on every run. A file
is written only if it does not already hold exactly these bytes (write_text): the texts are the same under both truths,
and a raw run and a coarse run side by side would otherwise each replace files that the other has open, which Windows
refuses. They are read by the pipeline's own loader like any config: the benchmark is only worth anything if it scores
what the submission pipeline does. The writer handles the flat tables that a Config holds, str, int, float and bool, and
refuses a parameter of another kind by name (toml_value) before any file is written.

--variants picks others, or a subset, by suffix (VARIANT_NAMES: the five above, then restart and restart_run, then the six plus_ ones). The two
restart variants replace nothing: <stem>__restart is CONFIG's evidence plus oracle_restart at weight 1.0, and
<stem>__restart_run the same with oracle_restart_run: what a perfect "which team restarts" source would be worth
(P17). Only a variant that replaces the home source needs one, so a config with none (main.toml) can have the
restart variants and is refused the others, by name. Without --variants it is the classic five, as always.

The plus_ variants (PLUS_VARIANTS) are the home variants that replace nothing: <stem>__plus_home_visible, __plus_home_gaps
and __plus_home_all are CONFIG's evidence with the oracle appended at weight 1.0, and __alive_plus_home_visible, _gaps
and _all the same with oracle_alive added first. They are for a config with several home sources (main.toml has
controlled_touch and shape_possession and no nearest_player), which the classic variants refuse because there is no
single source to replace. Adding is close to replacing where the oracle speaks, and the oracle is the better measure of
what is left: it carries about 13.8 nats a frame at eps 1e-6 and weight 1, against at most about 2.6 for any real source
the config has, so the decoder follows it; where it is silent (NaN) the config's own sources decide, which is what
plus_home_gaps and plus_home_visible each leave to them on the other side of the ball's presence. A config's own sources
stay in the decode as they are, so the headroom is that of this config as it is. A source that is itself hard (a
nearest_player with confidence 1.0 is 13.8 nats too) ties with the oracle where it is wrong, and the variant falls short
of the truth by that much.

Reports are written only with --write (write=True), for the five variants and not for CONFIG: <stem>__<variant>.json
under <data dir>/runs/benchmark/<profile>/, with the suffixes of the runner's own rule for a subset of matches, the
coarse truth and a number of seeds. CONFIG's report is the runner's to write (run.py), where it can say what a headroom
run cannot, which other configs it was compared with; a report of CONFIG written here would replace that one and lose
its `versus` block.

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
# The kinds of source that say who has the ball, one of which the home variants replace. A new one (P04's, when it
# lands) is added here, or the variants cannot replace it.
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
# P17's restart oracles: (suffix of the stem, oracle added to the config's own evidence, what it measures). They replace
# nothing, so a config with no home source (main.toml has none) can have them.
RESTART_TAIL = 50           # frames after a restart that oracle_restart speaks on (2 s at 25 fps)
RESTART_VARIANTS = (
    ("restart", "oracle_restart",
     "a perfect restart team: the team that restarts after each stoppage, for the first {} frames of its possession "
     "(oracle_restart added)".format(RESTART_TAIL)),
    ("restart_run", "oracle_restart_run",
     "a perfect restart team held for the whole possession that follows (oracle_restart_run added)"),
)
# The plus_ variants: (suffix of the stem, alive oracle added, home oracle added, what it measures). They replace nothing,
# the oracle is appended after the config's own sources (and after oracle_alive), so a config with several home sources,
# none of them nearest_player (main.toml), can have them.
PLUS_VARIANTS = (
    ("plus_home_visible", False, "oracle_home_visible",
     "a perfect possessor where the ball is seen, added to the config's own sources (oracle_home_visible added)"),
    ("plus_home_gaps", False, "oracle_home_gaps",
     "a perfect possessor only where the ball is missing, added to the config's own sources (oracle_home_gaps added)"),
    ("plus_home_all", False, "oracle_home_all",
     "a perfect possessor on every live frame, added to the config's own sources (oracle_home_all added)"),
    ("alive_plus_home_visible", True, "oracle_home_visible",
     "a perfect dead-ball detector and a perfect possessor where the ball is seen, both added"),
    ("alive_plus_home_gaps", True, "oracle_home_gaps",
     "a perfect dead-ball detector and a perfect possessor where the ball is missing, both added"),
    ("alive_plus_home_all", True, "oracle_home_all",
     "both, whole, added to the config's own sources: where an oracle speaks it outweighs them, so about the truth"),
)
VARIANT_NAMES = (tuple(suffix for suffix, *_ in VARIANTS) + tuple(suffix for suffix, *_ in RESTART_VARIANTS)
                 + tuple(suffix for suffix, *_ in PLUS_VARIANTS))
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


def restart_home(codes, graded, tail):
    """home of the restart oracle: 1.0 / 0.0 for the team that restarts after each stoppage, NaN elsewhere.

    Within each block (half, start, stop) of `graded`, take each maximal run of D, [a, e), with e < stop and codes[e]
    H or A: the team in codes[e] restarts. Let f be the first frame after e whose code differs from codes[e] (or stop):
    home[e:min(e + tail, f)] is 1.0 if that team is H, else 0.0. `tail` is a number of frames, at least 1, or None for
    the whole possession, up to f. A D run at a block's start counts (it is the kick-off); one followed by X, or one
    that reaches stop, restarts nothing. Read-only float64, of len(codes).
    """
    if tail is not None and (isinstance(tail, bool) or not isinstance(tail, int) or tail < 1):
        raise ValueError("restart_home: tail must be a whole number of frames, at least 1, or None, got {!r}".format(
            tail))
    codes = np.asarray(codes)
    home = no_opinion(len(codes))
    for _, start, stop in graded:
        dead = np.concatenate(([False], codes[start:stop] == D, [False]))
        edges = np.flatnonzero(dead[1:] != dead[:-1])           # a D run is [edges[2k], edges[2k+1]) in the block
        for end in edges[1::2] + start:
            if end >= stop or codes[end] not in (H, A):
                continue
            team = codes[end]
            different = np.flatnonzero(codes[end:stop] != team)
            following = end + (int(different[0]) if len(different) else stop - end)
            home[end:following if tail is None else min(end + tail, following)] = 1.0 if team == H else 0.0
    home.setflags(write=False)
    return home


def oracles(codes_of):
    """{kind: compute} for the six oracle sources, reading the truth from `codes_of`.

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

    def restart_oracle(kind, tail):
        def compute(run_context):
            codes = truth_of(kind, run_context)
            return Evidence(alive=no_opinion(len(codes)), home=restart_home(codes, run_context.inputs.graded, tail))
        return compute

    def home_oracle(kind, where):
        # where: "visible" (the ball is present), "gaps" (it is missing) or None (everywhere)
        def compute(run_context):
            codes = truth_of(kind, run_context)
            speaks = (codes == H) | (codes == A)
            if where == "visible":
                speaks &= run_context.inputs.atd.ball_present
            elif where == "gaps":
                speaks &= ~run_context.inputs.atd.ball_present
            home = no_opinion(len(codes))
            home[speaks] = np.where(codes[speaks] == H, 1.0, 0.0)
            return Evidence(alive=no_opinion(len(codes)), home=home)
        return compute

    return {ALIVE_ORACLE: oracle_alive,
            "oracle_home_visible": home_oracle("oracle_home_visible", "visible"),
            "oracle_home_gaps": home_oracle("oracle_home_gaps", "gaps"),
            "oracle_home_all": home_oracle("oracle_home_all", None),
            "oracle_restart": restart_oracle("oracle_restart", RESTART_TAIL),
            "oracle_restart_run": restart_oracle("oracle_restart_run", None)}


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

def variant_specs(variants=None):
    """[(suffix, alive oracle, home oracle replacing the source, oracle added, meaning)] of the variants asked for.

    None is the classic five, in their order; otherwise the names of VARIANT_NAMES, in the order given. BenchmarkError
    for a name that is not one, or one given twice.
    """
    classic = {suffix: (suffix, alive, home, None, meaning) for suffix, alive, home, meaning in VARIANTS}
    restart = {suffix: (suffix, False, None, kind, meaning) for suffix, kind, meaning in RESTART_VARIANTS}
    plus = {suffix: (suffix, alive, None, kind, meaning) for suffix, alive, kind, meaning in PLUS_VARIANTS}
    if variants is None:
        return list(classic.values())
    chosen = []
    for suffix in variants:
        if suffix not in VARIANT_NAMES:
            raise run.BenchmarkError("variant {!r} is not one of: {}".format(suffix, ", ".join(VARIANT_NAMES)))
        if suffix in [name for name, *_ in chosen]:
            raise run.BenchmarkError("variant {!r} is asked for twice".format(suffix))
        chosen.append(classic.get(suffix) or restart.get(suffix) or plus[suffix])
    return chosen


def check_config(config, variants=None):
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
    if any(home is not None for _, _, home, _, _ in variant_specs(variants)):
        home_source(config)     # only a variant that replaces it needs one: the restart and plus_ variants add


def home_source(config):
    """The evidence Component that the home oracles replace; BenchmarkError unless `config` has exactly one."""
    found = [component for component in config.evidence if component.kind in HOME_KINDS]
    if not found:
        raise run.BenchmarkError(
            "{}: no evidence source of kind {} to replace with the home oracles (the variants home_visible, home_all, "
            "alive_home_visible and alive_home_all take its place); add one, or run a config that has it (a new kind "
            "of source that says who has the ball is added to HOME_KINDS in pipeline/benchmark/headroom.py)".format(
                config.stem, " or ".join(HOME_KINDS)))
    if len(found) > 1:
        raise run.BenchmarkError(
            "{}: {} evidence sources of kind {} ({}); the home oracles replace one, and which one is not for the "
            "runner to say; keep one".format(config.stem, len(found), " or ".join(HOME_KINDS),
                                             ", ".join(repr(component.name) for component in found)))
    return found[0]


def variant_evidence(config, alive, home, extra=None):
    """The evidence of a variant: `config`'s own, the home source replaced by the oracle `home` (None: it stays) in its
    place with its weight, ALIVE_ORACLE added after the rest if `alive`, and then the oracle `extra` if there is one."""
    entries = list(config.evidence)
    if home is not None:
        replaced = home_source(config)
        at = entries.index(replaced)
        entries[at] = Component(kind=home, name=home, weight=replaced.weight, params={})
    if alive:
        entries.append(Component(kind=ALIVE_ORACLE, name=ALIVE_ORACLE, weight=1.0, params={}))
    if extra is not None:
        entries.append(Component(kind=extra, name=extra, weight=1.0, params={}))
    return tuple(entries)


def variant_texts(config, variants=None):
    """[(stem, TOML text)] of the variants of `config` (default the classic five), in the order they run. BenchmarkError
    if any cannot be made (no home source, a parameter that cannot be written), so that nothing has been written when
    it is raised."""
    source = run.shown(config.path)
    texts = []
    for suffix, alive, home, extra, meaning in variant_specs(variants):
        stem = "{}__{}".format(config.stem, suffix)
        note = ["Written by pipeline/benchmark/headroom.py from {}, and rewritten by every run: edit that, not "
                "this.".format(source),
                "Variant {}: {}.".format(suffix, meaning),
                "Its oracle sources read the benchmark's truth and exist only while headroom.py runs. Not a config to "
                "submit."]
        description = "{}Headroom variant of {}: {}.".format(config.description + " " if config.description else "",
                                                             config.stem, meaning)
        texts.append((stem, toml_text(config, variant_evidence(config, alive, home, extra), description, note)))
    return texts


def write_variants(config, data_dir=None, variants=None):
    """Write the variants of `config` into <data dir>/runs/benchmark/headroom/configs/ (data dir default
    games.DATA_DIR); [(stem, path)] in the order they run. Every text is made before the first file is written."""
    folder = os.path.join(data_dir or games.DATA_DIR, *FOLDER)
    texts = variant_texts(config, variants)
    written = []
    for stem, text in texts:
        path = os.path.join(folder, stem + ".toml")
        write_text(path, text)
        written.append((stem, path))
    return written


def write_text(path, text):
    """`text` at `path`, UTF-8 with LF endings; True if written, False if the file already held exactly these bytes.

    Atomically: through a temporary file in the folder, renamed over the old one, so that a run that is interrupted
    never leaves half a config for the next to load. A file that already holds the bytes is not touched at all, not even
    its modification time: the variants of a config are the same text under the raw and the coarse truth, two runs side
    by side write the same files, and on Windows os.replace onto a file that another process has open (the other run,
    reading its configs) is a PermissionError. The same error where the file turns out to hold these bytes by then is
    the other run having written them first, which is as good as writing them: it is not raised. Any other one is.
    """
    data = text.encode("utf-8")
    if _holds(path, data):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        try:
            os.replace(temporary, path)
        except PermissionError:
            if not _holds(path, data):
                raise
            os.remove(temporary)
            return False
        return True
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


def _holds(path, data):
    """True if the file at `path` exists and holds exactly `data`."""
    try:
        with open(path, "rb") as handle:
            return handle.read() == data
    except OSError:
        return False


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
                 calibration=None, data_dir=None, log=print, write=False, variants=None, drop_match_only=False):
    """Run CONFIG and its variants (the classic five, or `variants`, names from VARIANT_NAMES) through evaluate_configs;
    its result (CONFIG is the reference).

    The arguments are evaluate_configs' own, with `write`: True writes the report of each variant (not CONFIG's: the
    reference is the runner's to write, run.py) as run.write_reports does, and logs where. The variants are written
    first into <data dir>/runs/benchmark/headroom/configs/ (write_variants), always, since the runner reads them as
    configs, and only where the file does not already hold the same bytes; `log` receives a line for each, then the
    runner's progress and table. `matches` is {id: DflMatch} to use instead of loading (the checks pass synthetic ones);
    its truth is made at once. `drop_match_only` is the runner's: each config, the variants too, runs without its
    match-only sources (main.toml's Smart Tagging), which the benchmark cannot serve.

    What is refused, and when. ConfigError for a config that does not load, and BenchmarkError for one that cannot be
    run or have variants made of it (scope = "file", a dot in its name, no or several nearest_player sources, a
    parameter that cannot be written, an unknown truth): all of these before any file is written. What only the
    runner knows is refused by it, after the variants are written, because it loads them: an unknown profile or match,
    a repeated match, a number of seeds below 1, a calibration that does not suit the profile. The files are a few
    small configs in a fixed folder, rewritten by the next run, and no report is written for a run that was refused.
    """
    if truth not in run.TRUTHS:
        raise run.BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(run.TRUTHS)))
    config = load_config(config_path)
    check_config(config, variants)
    written = write_variants(config, data_dir, variants)

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
    for (stem, _), (*_, meaning) in zip(written, variant_specs(variants)):
        log("variant   {:<{}}  {}".format(stem, width, meaning))
    with registered(evidence_stage.REGISTRY, **oracles(codes_of)):
        result = run.evaluate_configs([config.path] + [path for _, path in written], profile=profile, seeds=seeds,
                                      match_ids=match_ids, matches=served, calibration=calibration, truth=truth,
                                      data_dir=data_dir, log=log, drop_match_only=drop_match_only)
    if write:
        # The variants' reports only: CONFIG is the reference, and a report of it written here would replace the
        # runner's own, which has the `versus` block of the run that it was in.
        log("")
        for path in run.write_reports(dict(result, configs=result["configs"][1:])):
            log("report    {}".format(run.shown(path)))
    return result


# -- the command line -------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run CONFIG on the benchmark beside variants of it whose evidence is the truth (an oracle "
                    "dead-ball detector, an oracle possessor, both): what a perfect source would be worth to it. "
                    "CONFIG is the reference of the comparisons; the classic home variants need it to have one "
                    "nearest_player source, the restart and plus_ ones do not.")
    parser.add_argument("config", metavar="CONFIG",
                        help="a TOML config (a nearest_player source is needed only for the classic variants), e.g. "
                             "pipeline/benchmark/configs/nearest_w50.toml")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--truth", choices=run.TRUTHS, default="raw",
                        help="the truth to score against, and that the oracles carry: raw, or coarse")
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--variants", nargs="+", metavar="SUFFIX", choices=VARIANT_NAMES,
                        help="the variants to run, of {} (default: the first five; the restart and plus_ ones need "
                             "no nearest_player)".format(", ".join(VARIANT_NAMES)))
    parser.add_argument("--drop-match-only", action="store_true",
                        help="run CONFIG and its variants without their match-only sources (as run.py's flag does), "
                             "e.g. main.toml's smart_tagging, which a DFL match cannot serve")
    parser.add_argument("--write", action="store_true",
                        help="also write the reports of the variants, not CONFIG's (default: print only)")
    args = parser.parse_args(argv)
    try:
        run_headroom(args.config, profile=args.profile, seeds=args.seeds, truth=args.truth, match_ids=args.matches,
                     data_dir=args.data_dir, write=args.write, variants=args.variants,
                     drop_match_only=args.drop_match_only)
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
