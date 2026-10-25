"""The walk of exported boosted trees, shared by the learned evidence sources: numpy only, no pickle, no scikit-learn.

A learned source is trained under pipeline/benchmark/ (scikit-learn lives there and nowhere else), written as JSON
under pipeline/models/, and run here with numpy alone, so the submission pipeline's pins stay numpy and pandas and no
pickle is ever loaded (a pickle would run code from the file; JSON can only be numbers). `dead_ball_model` was the
first such source; this module holds what any of them needs to run the trees of a gradient-boosted classifier, so that
two sources walk them the same way, to the bit.

The format. A model is a dict with a numeric `baseline` and a list `trees`; a tree holds seven node arrays of one
length, indexed by node, node 0 being the root: `feature`, `threshold`, `missing_left`, `left`, `right`, `leaf` and
`value`. A row goes left at a split where x <= threshold, or where x is NaN and `missing_left` is set, and right
otherwise; it stops at a node whose `leaf` is set, and the model's raw score is the baseline plus the `value` of its
leaf in every tree, added in tree order, which is scikit-learn's own order, so the scores agree with `predict_proba`
(through the sigmoid) to rounding; the trainer's check holds them to 1e-12. NaN features go where the trees were
fitted to send them: the tracker's missing ball is the commonest feature value of the match. A threshold of +inf
(scikit-learn's split on missingness) is stored as the largest float, which is exact for finite features, so a
feature that is infinite is outside what the walk can promise: the caller refuses it (dead_ball_model does).

`check_trees` is the load-time check of one model: the arrays are there and as long as each other, a tree is small
enough for the walk, and every split names a feature and children that come later. A child always has a larger index
than its parent, which the walk relies on, so the check is what makes it safe to run a file nobody here wrote.
`score` is the walk. `who` names the caller in the messages, so a source's errors start with its own name.

The speed. Walking 300 trees over 140,000 rows is where the time goes, so the walk is made of whole-array operations
on bytes and not of a loop over rows or of row subsets (an index-subset walk, tried first, cost about twice as much).
A tree is read bottom up on all the rows of a chunk (`CHUNK` rows, so that the byte arrays of a step stay in the
cache): a node's leaf id is its left child's where the row goes left and its right child's where not, so one pass from
the last node to the root gives every row its leaf. The choice between the two ids is made with the byte masks 255 (go
left) and 0: the smaller of (left id, mask) is the id or 0, the smaller of (right id, 255 - mask) the other way round,
and the larger of the two is the chosen one (`np.where` does the same and measured about 7% slower). A leaf's id is
its node index, so a tree has at most `MAX_NODES` nodes (one of 15 leaves has 29; a bigger one is refused on load).
The leaf values are then read from the tree's own table and added to the raw score of every row, in tree order.
"""

import numbers

import numpy as np

# Params
CHUNK = 32768                                          # rows per step of the tree walk
MAX_NODES = 255                                        # a leaf id is a node index held in one byte, and 255 is a mask


def check_trees(model, n_features, where, who):
    """ValueError, starting with `who`, unless `model` (a dict) holds trees the walk can run on `n_features` columns.

    `where` says which model of which file ("full of pipeline/models/x.json") and goes into the messages.
    """
    if not isinstance(model.get("baseline"), numbers.Real) or not isinstance(model.get("trees"), list):
        raise ValueError("{}: model {} needs a numeric `baseline` and a list of `trees`".format(who, where))
    for number, tree in enumerate(model["trees"]):
        parts = [tree.get(key) for key in ("feature", "threshold", "missing_left", "left", "right", "leaf", "value")]
        if any(not isinstance(part, list) for part in parts) or len({len(part) for part in parts}) != 1 or not parts[0]:
            raise ValueError("{}: tree {} of {} must hold seven node arrays of one length"
                             .format(who, number, where))
        feature, left, right, leaf = parts[0], parts[3], parts[4], parts[5]
        if len(feature) > MAX_NODES:
            raise ValueError("{}: tree {} of {} has {} nodes and the walk reads at most {}; retrain with "
                             "fewer leaves".format(who, number, where, len(feature), MAX_NODES))
        for node in range(len(feature)):
            if leaf[node]:
                continue
            if not 0 <= feature[node] < n_features or not (node < left[node] < len(feature)
                                                           and node < right[node] < len(feature)):
                raise ValueError("{}: tree {} of {} has a split at node {} that names no feature or "
                                 "no later child".format(who, number, where, node))


def score(model, X):
    """The raw score of every row of the 2-D X: the baseline plus the leaf of every tree, CHUNK rows at a time.

    `model` has passed `check_trees` for X's number of columns. The sum is made tree by tree in tree order, so a row's
    score is the same whatever the chunk it falls in.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("trees: X must be 2-D, (rows, features), got shape {}".format(X.shape))
    return np.concatenate([_chunk(model, X[start:start + CHUNK]) for start in range(0, len(X), CHUNK)]
                          or [np.empty(0)])


def _chunk(model, X):
    """baseline + the leaf of every tree, added tree by tree, so a row's sum is made in scikit-learn's own order."""
    raw = np.full(len(X), float(model["baseline"]))
    columns = np.asfortranarray(X)                       # a column is one contiguous run: x = columns[:, f]
    missing = {}                                         # feature -> its NaN mask, found only if a node needs it
    ids = {}                                             # leaf id -> that id on every row (np.minimum with a scalar is
                                                         # several times slower than with an array)
    with np.errstate(invalid="ignore"):                  # NaN <= threshold is False, which is what is wanted
        for tree in model["trees"]:
            feature, threshold, missing_left = tree["feature"], tree["threshold"], tree["missing_left"]
            left, right, leaf = tree["left"], tree["right"], tree["leaf"]
            if leaf[0]:
                raw += tree["value"][0]
                continue
            reach = [None] * len(feature)                # per node: the leaf id of every row that starts there
            for node in range(len(feature) - 1, -1, -1):
                if leaf[node]:
                    if node not in ids:
                        ids[node] = np.full(len(X), node, dtype=np.uint8)
                    reach[node] = ids[node]
                    continue
                goes_left = columns[:, feature[node]] <= threshold[node]
                if missing_left[node]:
                    if feature[node] not in missing:
                        missing[feature[node]] = np.isnan(columns[:, feature[node]])
                    goes_left |= missing[feature[node]]
                to_left = np.negative(goes_left.view(np.uint8))              # 255 where the row goes left, else 0
                to_right = np.invert(to_left)                                # and the other way round
                reach[node] = np.maximum(np.minimum(reach[left[node]], to_left),
                                         np.minimum(reach[right[node]], to_right))
            raw += np.array(tree["value"], dtype=np.float64)[reach[0]]
    return raw
