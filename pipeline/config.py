"""Read a run's TOML config, strictly, before any data is touched.

A config says which components run and with which parameters; pipeline/configs/ holds them, and the file name is
the run's name -- pipeline/configs/main.toml writes evaluation/results/main.csv. pipeline/README.md has the schema.

Strict, because a silently ignored typo is the worst bug an experiment can have: `half_widht = 75` would run the
default, score it, and get logged as a result about 75. So every key is checked -- the top level against a fixed
list, each component's parameters against the signature of the function they will be passed to -- and all of it
happens here, before the ATD cache is loaded, so a mistake costs a second rather than a minute.

What is not checked is the *type* of a parameter (half_width = "50" binds fine). Components validate their own.

The optional [inputs] table says what the run reads besides the tactical tracking. Its one key, ball_fill, is "none"
(the default: our ball only, as every config before R028) or "broadcast" (our ball's gaps filled from the broadcast's,
Inputs.with_ball_fill); a value or a key that is not one of those stops the run here too.
"""

import inspect
import os
import tomllib
from dataclasses import dataclass

# Params
TOP_LEVEL_KEYS = ("description", "seed", "scope", "inputs", "evidence", "decoder", "refine")
SCOPES = ("halves", "file")
DEFAULT_SCOPE = "halves"
DEFAULT_SEED = 0
INPUTS_KEYS = ("ball_fill",)           # the keys an [inputs] table may hold
BALL_FILLS = ("none", "broadcast")     # [inputs] ball_fill: what fills our ball's gaps (R028)
DEFAULT_BALL_FILL = "none"
# Keys of an [[evidence]] entry that belong to the runner rather than the component; everything else is a parameter.
EVIDENCE_RESERVED = ("kind", "name", "weight")
# How many leading positional arguments each kind of component takes before its keyword-only parameters:
# compute(run), decode(ctx), refine(ctx, decoded). Used to check a config's parameters against the signature.
CONTEXT_ARGUMENTS = {"evidence": 1, "decode": 1, "refine": 2}


class ConfigError(ValueError):
    """A config that cannot run. The message names the file, the entry and what to change."""


@dataclass(frozen=True)
class Component:
    """One [[evidence]], [decoder] or [[refine]] entry: which function, under what name, with which parameters."""
    kind: str      # key in that stage's REGISTRY
    name: str      # unique among evidence sources; defaults to kind
    weight: float  # evidence only: how much the decoder trusts the source (1.0 for decoders and refiners)
    params: dict   # passed to the function as keyword arguments


@dataclass(frozen=True)
class Config:
    path: str          # absolute path of the TOML file
    stem: str          # file name without .toml -- the run's name, and its output's
    text: str          # the file as written, kept for the run manifest
    description: str
    seed: int
    scope: str         # "halves": one block per graded half; "file": the whole file as one block
    ball_fill: str     # [inputs] ball_fill: "none", our ball only; "broadcast", its gaps filled from the broadcast's
    evidence: tuple    # Component, in config order
    decoder: Component
    refine: tuple      # Component, applied in order


def default_registries():
    """Every stage's REGISTRY, imported here rather than at module level so the stages can import config freely."""
    from pipeline import decode, evidence, refine
    return {"evidence": evidence.REGISTRY, "decode": decode.REGISTRY, "refine": refine.REGISTRY}


def load_config(path, registries=None):
    """Parse and check a config file. Returns a Config, or raises ConfigError saying what to fix.

    registries maps "evidence" / "decode" / "refine" to {kind: function}; the default is the real stages. The
    self-test passes stand-ins, so the checks here can be tested without any real component.
    """
    path = os.path.abspath(path)
    where = os.path.basename(path)
    registries = registries or default_registries()
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        table = tomllib.loads(text)
    except OSError as error:
        raise ConfigError("cannot read config {}: {}".format(path, error))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError("{} is not valid TOML: {}".format(where, error))

    unknown = sorted(set(table) - set(TOP_LEVEL_KEYS))
    if unknown:
        raise ConfigError("{}: unknown top-level key(s) {}; allowed: {}".format(
            where, ", ".join(unknown), ", ".join(TOP_LEVEL_KEYS)))

    scope = table.get("scope", DEFAULT_SCOPE)
    if scope not in SCOPES:
        raise ConfigError("{}: scope {!r} is not one of: {}".format(where, scope, ", ".join(SCOPES)))
    seed = table.get("seed", DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("{}: seed must be an integer".format(where))
    description = table.get("description", "")
    if not isinstance(description, str):
        raise ConfigError("{}: description must be a string".format(where))
    ball_fill = _ball_fill(table.get("inputs", {}), where)

    evidence_entries = _table_list(table, "evidence", where)
    if not evidence_entries:
        raise ConfigError("{}: needs at least one [[evidence]] entry".format(where))
    evidence = tuple(_component(entry, "evidence", index, registries["evidence"], where)
                     for index, entry in enumerate(evidence_entries, start=1))
    names = [component.name for component in evidence]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigError("{}: evidence name {} is used twice; give one of them `name = \"...\"`".format(
            where, ", ".join(repr(name) for name in repeated)))

    decoder_entry = table.get("decoder")
    if not isinstance(decoder_entry, dict):
        raise ConfigError("{}: needs exactly one [decoder] table".format(where))
    decoder = _component(decoder_entry, "decode", None, registries["decode"], where)

    refine = tuple(_component(entry, "refine", index, registries["refine"], where)
                   for index, entry in enumerate(_table_list(table, "refine", where), start=1))

    return Config(path=path, stem=os.path.splitext(os.path.basename(path))[0], text=text,
                  description=description, seed=seed, scope=scope, ball_fill=ball_fill,
                  evidence=evidence, decoder=decoder, refine=refine)


def _ball_fill(inputs, where):
    """[inputs] ball_fill, checked; DEFAULT_BALL_FILL when the table or the key is not there."""
    if not isinstance(inputs, dict):
        raise ConfigError("{}: inputs must be written as one [inputs] table, e.g. ball_fill = \"broadcast\"".format(
            where))
    unknown = sorted(set(inputs) - set(INPUTS_KEYS))
    if unknown:
        raise ConfigError("{}: [inputs] has unknown key(s) {}; allowed: {}".format(
            where, ", ".join(unknown), ", ".join(INPUTS_KEYS)))
    ball_fill = inputs.get("ball_fill", DEFAULT_BALL_FILL)
    if ball_fill not in BALL_FILLS:
        raise ConfigError("{}: [inputs] ball_fill {!r} is not one of: {}".format(
            where, ball_fill, ", ".join(BALL_FILLS)))
    return ball_fill


def _table_list(table, key, where):
    """[[key]] entries as a list of dicts; [] when absent. Catches the easy slip of writing [key] instead."""
    entries = table.get(key, [])
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ConfigError("{}: {} must be written as [[{}]] tables (double brackets), one per entry".format(
            where, key, key))
    return entries


def _component(entry, stage, index, registry, where):
    """Check one entry against its stage's registry and its function's signature, and build the Component."""
    label = "[decoder]" if index is None else "[[{}]] #{}".format(stage, index)
    kind = entry.get("kind")
    if not isinstance(kind, str):
        raise ConfigError("{}: {} needs kind = \"...\"".format(where, label))
    if kind not in registry:
        raise ConfigError("{}: {} has no component {!r}; available: {}".format(
            where, label, kind, ", ".join(sorted(registry)) or "none yet -- this stage's REGISTRY is empty"))

    reserved = EVIDENCE_RESERVED if stage == "evidence" else ("kind",)
    params = {key: value for key, value in entry.items() if key not in reserved}
    name = entry.get("name", kind) if stage == "evidence" else kind
    weight = entry.get("weight", 1.0) if stage == "evidence" else 1.0
    if not isinstance(name, str) or not name:
        raise ConfigError("{}: {} name must be a non-empty string".format(where, label))
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ConfigError("{}: {} weight must be a number >= 0".format(where, label))

    _check_parameters(registry[kind], stage, params, "{}: {} {!r}".format(where, label, kind))
    return Component(kind=kind, name=name, weight=float(weight), params=params)


def _check_parameters(function, stage, params, who):
    """ConfigError unless function(<context arguments>, **params) would bind.

    Unknown and missing names are reported by name first -- inspect's own message mentions only the first problem
    it meets, which for a typo is the *missing* correct spelling, never the wrong one that was typed.
    """
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())[CONTEXT_ARGUMENTS[stage]:]
    accepted = [parameter.name for parameter in parameters
                if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)]
    # A component that takes **kwargs opts out of this check, and has to validate its own parameters.
    takes_anything = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters)
    unknown = [] if takes_anything else sorted(set(params) - set(accepted))
    missing = [parameter.name for parameter in parameters
               if parameter.default is parameter.empty and parameter.name not in params
               and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)]
    problems = []
    if unknown:
        problems.append("unknown parameter(s) {}".format(", ".join(unknown)))
    if missing:
        problems.append("missing {}".format(", ".join(missing)))
    if not problems:
        try:
            # Placeholders for the context arguments, the config's keys for the rest: exactly the call the runner
            # will make, minus the work. Catches whatever the name checks above cannot, like positional-only.
            signature.bind(*([None] * CONTEXT_ARGUMENTS[stage]), **params)
        except TypeError as error:
            problems.append(str(error))
    if problems:
        raise ConfigError("{}: {}; it accepts: {}".format(
            who, "; ".join(problems), ", ".join(accepted) or "no parameters"))
