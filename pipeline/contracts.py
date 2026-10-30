"""The few types every stage agrees on -- the one part of the pipeline that is expensive to change.

Five people write components against these, so they are kept small and checked at every hand-over. A component is
a plain function; what it receives and returns is defined here and nowhere else:

    evidence   compute(run, **params)          -> Evidence    sees the whole match, gives one opinion per frame
    decoder    decode(ctx, **params)           -> Decoded     sees one block (a graded half), returns its labels
    refiner    refine(ctx, decoded, **params)  -> Decoded     sees one block, labels in, labels out

`**params` are the keys of the component's table in the config, passed as keyword arguments.

Why evidence is two numbers per frame and not three
---------------------------------------------------
The ground truth is built in two steps: D where the ball is dead, otherwise the team that has it (CLAUDE.md). Most
sources we plan to add answer only one of those questions -- a dead-ball detector (P05) says
nothing about which team, nearest-player control (P04) or team shape (P07) says nothing about whether play is
live. So a source reports

    alive   P(ball in play)                       NaN = no opinion
    home    P(home team has it | ball in play)    NaN = no opinion

and a decoder recombines them into the three scored labels:

    P(H) = alive * home     P(A) = alive * (1 - home)     P(D) = 1 - alive

A three-class vector would force a team-only source to invent a P(D) it does not have. A NaN is not the same as
0.5: it means "leave this frame to the other sources", and a decoder must treat it as missing, not as a coin flip.
"""

from dataclasses import dataclass, replace

import numpy as np

import possession_metrics as pm

# Params
# Label codes are the scorer's own, so a decoded array goes straight into possession_metrics.evaluate().
H, A, D, X = pm.H, pm.A, pm.D, pm.X
LETTERS = np.array(pm.LABELS)                                    # code -> letter, for writing the CSV
CODES = {letter: code for code, letter in enumerate(pm.LABELS)}  # letter -> code, for configs
CHANNELS = ("alive", "home")


def no_opinion(n_frames):
    """A channel that says nothing: NaN on every frame. The starting point of every evidence array."""
    return np.full(n_frames, np.nan)


@dataclass(frozen=True, eq=False)
class Evidence:
    """One source's opinion on every frame of the match.

    Both arrays are float64, one value per submission row (= tactical video frame), so a source computed from the
    broadcast timeline fails the length check rather than shifting everything (docs/traps.md T09). `name` and
    `weight` come from the config; the runner fills them in, so a component returns Evidence(alive=..., home=...).

    weight is how much the decoder trusts this source. For a probabilistic decoder (P06) it is meant as an exponent
    on the source's likelihood -- weight 0.5 halves its log-likelihood; the stand-in majority vote only
    approximates that with a weighted mean.
    """
    alive: np.ndarray
    home: np.ndarray
    name: str = ""
    weight: float = 1.0

    def window(self, start, stop):
        """The same evidence restricted to frames [start, stop). Views, not copies."""
        return replace(self, alive=self.alive[start:stop], home=self.home[start:stop])


def check_evidence(evidence, n_frames):
    """Raise ValueError, naming the source, unless `evidence` keeps the contract above."""
    if not isinstance(evidence, Evidence):
        raise ValueError("an evidence source must return an Evidence, got {}".format(type(evidence).__name__))
    who = "evidence {!r}".format(evidence.name)
    for channel in CHANNELS:
        values = getattr(evidence, channel)
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError("{}: {} must be a 1-D numpy array".format(who, channel))
        if len(values) != n_frames:
            raise ValueError(
                "{}: {} has {} frames, expected {} -- one value per tactical video frame (a ~164k length means "
                "the broadcast timeline was indexed, docs/traps.md T09)".format(who, channel, len(values), n_frames))
        if values.dtype != np.float64:
            raise ValueError("{}: {} is {}, expected float64 with NaN for no opinion".format(
                who, channel, values.dtype))
        known = values[~np.isnan(values)]
        # One comparison each way also rejects +-inf, which a log of a hard 0 or 1 tends to produce.
        if known.size and (known.min() < 0.0 or known.max() > 1.0):
            raise ValueError("{}: {} has values outside [0, 1]; it is a probability".format(who, channel))


@dataclass(frozen=True)
class Block:
    """A stretch of frames decoded on its own: one graded half, or the whole file for scope = "file".

    Half-open, like every interval in this repository: frames start .. stop - 1.
    """
    half: object  # 1 or 2; None when the whole file is one block
    start: int
    stop: int

    def __len__(self):
        return self.stop - self.start

    def take(self, array):
        """array[start:stop] -- the block's share of any per-frame array of the whole match."""
        return array[self.start:self.stop]


@dataclass(frozen=True)
class RunContext:
    """What every component of one run can see: the aligned inputs and the run's seed.

    Randomness must come from the seed (docs/traps.md T22). A component that needs a generator makes its own with
    np.random.default_rng([run.seed, <a fixed number of its own>]), so adding a component never shifts the random
    stream of another.
    """
    inputs: object  # pipeline.inputs.Inputs; not imported here, to keep this module free of data code
    seed: int


@dataclass(frozen=True)
class BlockContext:
    """What a decoder or refiner sees: one block, and every evidence source cut down to that block.

    evidence maps the config's source names to Evidence windows, in config order. A decoder that needs a specific
    source looks it up by name; one that pools them iterates. Nothing here reaches across the block's edges --
    that is how the per-half rule (docs/traps.md T11) is kept without every decoder having to remember it.
    run.inputs is still the whole match, for components that need positions; slice it with block.take().
    """
    run: RunContext
    block: Block
    evidence: dict


@dataclass(frozen=True, eq=False)
class Decoded:
    """A block's labels, as the scorer's codes (H, A, D, X = 0, 1, 2, 3), plus an optional posterior.

    posterior is (len(block), 3) float64 -- P(H), P(A), P(D) per frame -- for decoders that have one (P06). It is
    what tells a refiner where an X would be free (inside a possession) and where it would be expensive (at a
    hand-over); a decoder without one leaves it None.
    """
    labels: np.ndarray
    posterior: object = None


def check_decoded(decoded, block, who):
    """Raise ValueError, naming the component, unless `decoded` holds valid labels for exactly this block."""
    if not isinstance(decoded, Decoded):
        raise ValueError("{} must return a Decoded, got {}".format(who, type(decoded).__name__))
    labels = decoded.labels
    if not isinstance(labels, np.ndarray) or labels.ndim != 1 or len(labels) != len(block):
        raise ValueError("{}: labels must be a 1-D array of {} frames for block {}..{}".format(
            who, len(block), block.start, block.stop - 1))
    if labels.dtype != np.int8:
        raise ValueError("{}: labels are {}, expected int8 codes (contracts.H, A, D, X)".format(who, labels.dtype))
    unknown = sorted(set(np.unique(labels).tolist()) - {H, A, D, X})
    if unknown:
        raise ValueError("{}: label codes outside H, A, D, X (0-3): {}".format(who, unknown))
    posterior = decoded.posterior
    if posterior is not None and (not isinstance(posterior, np.ndarray) or posterior.shape != (len(block), 3)):
        raise ValueError("{}: posterior must be None or a ({}, 3) array".format(who, len(block)))
