"""Stage 4 -- refine: post-processing of one decoded block, applied in config order. Empty on purpose.

A refiner is a function `refine(ctx, decoded, **params) -> Decoded`. It receives the same BlockContext the decoder
got -- one block, every evidence source cut to that block, and the whole match in ctx.run.inputs -- plus the
block's current labels, and returns new labels for the same frames. The runner checks every result the way it checks
the decoder's, so a refiner cannot change the length of a block or invent a label code.

    [[refine]]
    kind = "..."         # key in REGISTRY below
    # every other key is passed as a keyword argument

Nothing is registered yet, because nothing has been measured yet: every rule that belongs here trades one kind of
error for another (docs/traps.md T03-T05), and the right setting has to come from validation (P01), not from a
guess. pipeline/refine/README.md lists the first candidates -- P12's metric-aware clean-ups and P11's boundary
snapping -- and what each must be checked against.
"""

# Params
# kind (as written in a config) -> refine function. Starts empty; one line per refiner.
REGISTRY = {}
