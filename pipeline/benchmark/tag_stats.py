"""tag_stats: how far Smart Tagging's votes agree with our own pseudo-labels on BAR-ATH, in aggregates only (P08, R017).

    python pipeline/benchmark/tag_stats.py [--data-dir D]       # HEAVY: loads BAR-ATH, runs controlled_touch

    team = team_agreement(votes_team, touch_home, tags, graded=g, ball_missing=m)   # {"n", "share", "low", "high", ...}

R017 measured the dead votes (SET PIECES, GOALS) against dead_ball_model too: share 0.409 [0.280, 0.508],
lift 1.124, below the declared bar, so they were dropped and their measurement removed from here (recoverable from
cefb8b0).

Why. The tags are automatic phase intervals of our match only, not possessions and not ground truth (T17). There is no
truth to score them against and no benchmark (DFL has no tags), so the only honest bar is agreement with the two
pseudo-label of our own pipeline that is already trusted: the kept `controlled_touch` episodes (the rule of
`conventions.TOUCH`, confidence 0.99 and hold 0.5, so that `home` is a number exactly on the kept frames, the way
`train_shape.touch_agreement` takes them). The agreement becomes the votes' confidence,
rounded towards 0.5 to two decimals by P05's own helper (`train_dead_ball.towards_half`), and R017's declared bar
decided whether they enter the pipeline at all.

Team agreement is, among the graded frames inside a kept episode where the tags give a team vote (0 or 1: no vote and
disagreement are excluded), the share where the vote names the episode's team (`home` >= 0.5 is a home episode).
The bootstrap resamples whole tags with replacement, never frames: the frames of a tag are one phase and strongly
autocorrelated, so a frame-level interval would be far too narrow. A tag's pieces (a tag across both blocks has two) are
one tag. A frame covered by several tags is given to the first one that covers it (its votes agree there, or it is
not counted), so that nothing is counted twice. The interval is the 5th and 95th percentile of `draws` resampled
shares (a draw with no frame is dropped), from `default_rng(seed)`.

`tags` is smart_tagging.tags(inputs): [(kind, side, [(start, stop), ...]), ...], the tag-level form of `tag_spans`.

Nothing here prints a frame, a tag or a value per instance (CLAUDE.md, hard rules): the per-code table holds the code's
name, its frame count and its share. The CLI writes `<data dir>/runs/tag_stats/stats.json`, the same aggregates.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import conventions  # noqa: E402
from pipeline.benchmark.train_dead_ball import towards_half  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, smart_tagging  # noqa: E402

# Params
DRAWS = 2000
TEAM_SEED = (17, 0)             # default_rng(seed) of the team bootstrap, the design's
PERCENTILES = conventions.PERCENTILES
HOME_LEVEL = 0.5                # a touch's home >= this is a home episode
FOLDER = ("runs", "tag_stats")  # <data dir>/runs/tag_stats/stats.json


def rounded_towards_half(value):
    """`value` to two decimals towards 0.5, as P05's `--rules` rounds a rule's value (train_dead_ball.towards_half)."""
    return towards_half(value)


def _check_draws(draws):
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("tag_stats: draws must be a whole number of at least 1, got {!r}".format(draws))


def _check_length(name, array, n_frames):
    if np.shape(array) != (n_frames,):
        raise ValueError("tag_stats: {} must have one value per frame ({} of them), got shape {}".format(
            name, n_frames, np.shape(array)))


def _owner(tags, kind, n_frames):
    """(n_frames,) int: the index (among the tags of `kind`) of the first tag covering the frame, -1 where none."""
    owner = np.full(n_frames, -1, dtype=np.int64)
    chosen = [pieces for tag_kind, _, pieces in tags if tag_kind == kind]
    for index, pieces in enumerate(chosen):
        for start, stop in pieces:
            free = owner[start:stop] < 0
            owner[start:stop][free] = index
    return owner, len(chosen)


def _per_tag(owner, n_tags, eligible, hit):
    """(n, agreed): per tag, its eligible frames and those of them where `hit`."""
    mine = owner[eligible]
    n = np.bincount(mine[mine >= 0], minlength=n_tags)
    agreed = np.bincount(owner[eligible & hit][owner[eligible & hit] >= 0], minlength=n_tags)
    return n.astype(np.int64), agreed.astype(np.int64)


def _draws(n, agreed, draws, seed):
    """The shares of `draws` resamples of the tags (with replacement); resamples with no frame are dropped."""
    if not len(n):
        return np.array([])
    rng = np.random.default_rng(list(seed))
    pick = rng.integers(0, len(n), size=(draws, len(n)))
    total, hits = n[pick].sum(axis=1), agreed[pick].sum(axis=1)
    return hits[total > 0] / total[total > 0]


def _interval(shares):
    if not len(shares):
        return None, None
    low, high = np.percentile(shares, PERCENTILES)
    return float(low), float(high)


def _team_tally(team, touch_home, tags, graded):
    team, touch_home, graded = np.asarray(team), np.asarray(touch_home), np.asarray(graded, dtype=bool)
    n_frames = len(team)
    _check_length("touch_home", touch_home, n_frames)
    _check_length("graded", graded, n_frames)
    eligible = graded & ~np.isnan(touch_home) & ((team == 1) | (team == 0))
    hit = (team == 1) == (touch_home >= HOME_LEVEL)
    owner, n_tags = _owner(tags, "team", n_frames)
    n, agreed = _per_tag(owner, n_tags, eligible & (owner >= 0), hit)
    return eligible, hit, n, agreed


def team_draws(team, touch_home, tags, *, graded, draws=DRAWS, seed=TEAM_SEED):
    """The bootstrap shares of team_agreement (array of up to `draws` numbers), for the checks."""
    _check_draws(draws)
    _, _, n, agreed = _team_tally(team, touch_home, tags, graded)
    return _draws(n, agreed, draws, seed)


def team_agreement(team, touch_home, tags, *, graded, ball_missing, draws=DRAWS, seed=TEAM_SEED):
    """{"n", "share", "low", "high", "coverage", "coverage_gaps"}: the team votes against the kept touch episodes.

    `team` is smart_tagging.votes' team array, `touch_home` the controlled_touch `home` (NaN off the kept episodes),
    `tags` as smart_tagging.tags gives them, `graded` and `ball_missing` boolean per-frame masks. n counts the kept
    graded frames with a team vote of 0 or 1 and share (None if n is 0) those where it names the episode's team; low
    and high are the 90% tag-level bootstrap. coverage is the share of graded frames with such a vote, coverage_gaps
    the same over the graded frames whose ball is missing.
    """
    _check_draws(draws)
    team, graded = np.asarray(team), np.asarray(graded, dtype=bool)
    ball_missing = np.asarray(ball_missing, dtype=bool)
    _check_length("ball_missing", ball_missing, len(team))
    eligible, hit, n, agreed = _team_tally(team, touch_home, tags, graded)
    frames = int(eligible.sum())
    voted = graded & ((team == 1) | (team == 0))
    gaps = graded & ball_missing
    low, high = _interval(_draws(n, agreed, draws, seed))
    return {"n": frames, "share": float((eligible & hit).sum()) / frames if frames else None,
            "low": low, "high": high,
            "coverage": float(voted.sum()) / float(graded.sum()) if graded.any() else None,
            "coverage_gaps": float((voted & gaps).sum()) / float(gaps.sum()) if gaps.any() else None}


def per_code(inputs, touch_home):
    """[{"code", "frames", "share"}, ...] sorted by code: each team code's frames in the graded blocks and its own
    agreement, i.e. over its frames inside kept episodes, where its vote names the episode's team (None for a code
    with no frame to judge). Codes that cast no vote are left out. Several tags of a code add up (a frame two tags
    cover counts twice, being each tag's own frames). Aggregates only: no tag, frame or team value per instance."""
    touch_home = np.asarray(touch_home)
    totals = {}
    for tag in inputs.smart_tagging:
        kind, side = smart_tagging.tag_vote(tag)
        if kind != "team":
            continue
        row = totals.setdefault(str(tag["code"]).strip().upper(), [0, 0, 0])
        for _, block_start, block_stop in inputs.graded:
            start, stop = max(int(tag["start_frame"]), block_start), min(int(tag["end_frame"]), block_stop)
            if start >= stop:
                continue
            row[0] += stop - start
            kept = ~np.isnan(touch_home[start:stop])
            row[1] += int(kept.sum())
            row[2] += int((kept & ((touch_home[start:stop] >= HOME_LEVEL) == (side == 1))).sum())
    return [{"code": code, "frames": frames, "share": agreed / judged if judged else None}
            for code, (frames, judged, agreed) in sorted(totals.items())]


def _fmt(value):
    return "n/a" if value is None else "{:.3f}".format(value)


def render(team, codes):
    """The printed lines: the result, the per-code table and the confidence it would give (rounded towards 0.5).
    Figures and code names only."""
    c_team = None if team["share"] is None else rounded_towards_half(team["share"])
    lines = [
        "team     n {}  share {} [{}, {}]  coverage {}  coverage in ball gaps {}".format(
            team["n"], _fmt(team["share"]), _fmt(team["low"]), _fmt(team["high"]), _fmt(team["coverage"]),
            _fmt(team["coverage_gaps"])),
        "code                           frames   share"]
    lines += ["{:<30} {:>7}   {}".format(row["code"], row["frames"], _fmt(row["share"])) for row in codes]
    lines += ["would-be c_team {}   (towards 0.5, two decimals)".format(_fmt(c_team))]
    return lines


def measure(inputs):
    """(team, codes, document): the whole measurement of the team votes on `inputs`."""
    run_context = RunContext(inputs=inputs, seed=0)
    touch_home = controlled_touch.compute(run_context, **conventions.TOUCH).home
    team_votes = smart_tagging.votes(inputs)
    tags = smart_tagging.tags(inputs)
    graded = np.asarray(inputs.graded_mask, dtype=bool)
    ball_missing = np.isnan(np.asarray(inputs.atd.ball_xy)).any(axis=1)
    team = team_agreement(team_votes, touch_home, tags, graded=graded, ball_missing=ball_missing)
    codes = per_code(inputs, touch_home)
    document = {"team": team, "codes": codes, "draws": DRAWS,
                "c_team": None if team["share"] is None else rounded_towards_half(team["share"])}
    return team, codes, document


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Smart Tagging's agreement with our own pseudo-labels on BAR-ATH, in aggregates (HEAVY: loads the "
                    "real match).")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)

    from pipeline.benchmark import run
    from pipeline.inputs import Inputs
    try:
        inputs = Inputs(data_dir=args.data_dir)
        team, codes, document = measure(inputs)
        for line in render(team, codes):
            print(line)
        folder = os.path.join(args.data_dir or inputs.data_dir, *FOLDER)
        os.makedirs(folder, exist_ok=True)
        run.write_json(os.path.join(folder, "stats.json"), document)
        print("wrote     {}".format(os.path.join(folder, "stats.json")))
    except (run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
