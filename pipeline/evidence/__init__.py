"""Stage 2 -- evidence: independent per-frame opinions about the two questions the labels answer.

Every source is a function `compute(run, **params) -> Evidence` that sees the whole match (run.inputs) and returns,
for every frame, P(ball in play) and P(home has it | in play), with NaN meaning "no opinion here". The contract is
in pipeline/contracts.py; pipeline/evidence/README.md explains how to write a source.

To add one: write pipeline/evidence/<name>.py with a compute() function, add one line to REGISTRY below, and add an
[[evidence]] entry to a config. The runner checks every source's output -- length, dtype, range -- before any
decoder sees it.
"""

import numpy as np

from pipeline.contracts import A, D, H, X
from pipeline.evidence import ball_presence, controlled_touch, nearest_player

# Params
# kind (as written in a config) -> compute function. Explicit on purpose: one line per source, no import magic, and
# a merge conflict between two teammates' sources is a conflict on two adjacent lines.
REGISTRY = {
    "ball_presence": ball_presence.compute,
    "controlled_touch": controlled_touch.compute,
    "nearest_player": nearest_player.compute,
}


def describe(evidence, graded_mask, gt_codes=None):
    """Per source: how much of the graded match it has an opinion on and, given a ground truth, how often it agrees.

    Aggregates only -- shares and frame counts. Agreement reads a probability as a vote at 0.5 (>= 0.5 for home, as
    the decoder's tie rule does) and is measured only where both the source and the ground truth say something:

        home   on frames the GT calls H or A      (did the source name the right team?)
        alive  on frames the GT grades            (did it call dead where the GT says D?)

    On the public window that is a check that a source is wired up right -- nearest_player should agree with the
    GT on 77.8% of 5,640 frames (docs/results.md A01) -- not a measure of how good it is (docs/traps.md T01).
    """
    rows = []
    for source in evidence.values():
        row = {"name": source.name}
        for channel in ("alive", "home"):
            known = ~np.isnan(getattr(source, channel))
            row[channel + "_coverage"] = float(known[graded_mask].mean()) if graded_mask.any() else 0.0
        if gt_codes is not None:
            # home, on frames the GT calls H or A: did the source name the team that had the ball?
            judged = ~np.isnan(source.home) & ((gt_codes == H) | (gt_codes == A))
            row["home_judged_frames"] = int(judged.sum())
            row["home_agreement"] = (float(np.mean((source.home[judged] >= 0.5) == (gt_codes[judged] == H)))
                                     if judged.any() else None)
            # alive, on every frame the GT grades: did the source call the ball dead exactly where the GT says D?
            judged = ~np.isnan(source.alive) & (gt_codes != X)
            row["alive_judged_frames"] = int(judged.sum())
            row["alive_agreement"] = (float(np.mean((source.alive[judged] >= 0.5) == (gt_codes[judged] != D)))
                                      if judged.any() else None)
        rows.append(row)
    return rows
