"""Stage 3 -- decode: turn the evidence of one block (one graded half) into a label for every frame.

A decoder is a function `decode(ctx, **params) -> Decoded`. ctx.evidence holds every source cut down to ctx.block,
so a decoder cannot see across halftime even by accident (docs/traps.md T11); the runner calls it once per block and
writes X on every frame outside the blocks. The contract is in pipeline/contracts.py; pipeline/decode/README.md
explains the stage.

Exactly one decoder runs per config, chosen by [decoder] kind = "...". To add one: write
pipeline/decode/<name>.py with a decode() function and add one line to REGISTRY below.
"""

from pipeline.decode import majority_vote

# Params
# kind (as written in a config) -> decode function.
REGISTRY = {
    "majority_vote": majority_vote.decode,
}
