"""Stage 4 -- refine: post-processing of one decoded block, applied in config order.

A refiner is a function `refine(ctx, decoded, **params) -> Decoded`. It receives the same BlockContext the decoder
got -- one block, every evidence source cut to that block, and the whole match in ctx.run.inputs -- plus the
block's current labels, and returns new labels for the same frames. The runner checks every result the way it checks
the decoder's, so a refiner cannot change the length of a block or invent a label code.

    [[refine]]
    kind = "..."         # key in REGISTRY below
    # every other key is passed as a keyword argument

One refiner is registered: shift_edges (P02), which moves each kind of edge by a fixed number of frames and ships
with no value of its own (every default is 0, the identity). The others stay unwritten until something measured asks
for them: every rule that belongs here trades one kind of error for another (docs/traps.md T03-T05), and the right
setting has to come from validation (P01), not from a guess. pipeline/refine/README.md lists the candidates -- P12's
metric-aware clean-ups and P11's boundary snapping -- and what each must be checked against.
"""

from pipeline.refine import shift_edges

# Params
# kind (as written in a config) -> refine function. One line per refiner.
REGISTRY = {
    "shift_edges": shift_edges.refine,
}
