"""Checks of P27's two tools (pipeline/benchmark/p27_grid.py and p27_choose.py), which made R030's re-choice.

p27_grid writes a stage's configs; p27_choose applies to their reports the rule that the spec declared before the runs
(docs/superpowers/specs/2026-11-09-p27-degradation-design.md) and prints, on its last line, the arguments of the next
p27_grid call, which the chain scripts read. main.toml's constants are what the two said (R030, docs/results.md), and
two of the choices turned on the edge of the tie band: Stage A's h60 was inside it by 0.000004 and Stage C's
t06_mp025_w300 outside it by 0.000034. So what is held against the tools here is that edge first, with R030's own
figures, and then each declared rule in turn:

* the band is compared on the unrounded scores: 0.004996 below the best mean is tied and can win on the larger
  min(raw, coarse), 0.005034 below is not, whatever its min (test_p27_tie_band_edge_is_compared_on_the_unrounded_
  scores); exactly 0.005 below is pinned as it is today, decided by the floats (test_p27_a_mean_exactly_the_band_
  below_is_decided_by_the_floats);
* the grids and the incumbent are the spec's (test_p27_grids_and_incumbent_are_the_declared_ones); Stage A's and Stage
  C's stems are named by p27_choose exactly as p27_grid writes them, in the declared tie orders, each stem holding the
  setting that the choice reads from it (test_p27_stage_a_stems_..., test_p27_stage_c_stems_...), and an exact tie
  goes to that order (test_p27_an_exact_tie_goes_to_the_stages_tie_order);
* Stage D: the eleven configs, each group chosen by the same rule with the incumbent one candidate among the others,
  the groups' tie orders, the shape source, the arguments returned and the groups counted (test_p27_stage_d_*);
* the shift's guard and the final bar (test_p27_guard_*, test_p27_bar_*);
* p27_grid's configs: each loads as the runner loads it, the shift, the shape source, the confidence read from the
  touch file, the prefix (test_p27_grid_*);
* p27_choose's exit codes and its NEXT line, with R030's figures on the whole grids (test_p27_choose_main_*).

Everything is made up. A report is fake_report's: the fields that choose reads of a report of the generic profile's
full run (the seven DFL match ids, three seeds, no Calibration), each match being one half of UNIT true and UNIT
predicted possessions of which a given count is matched at every IoU. A pooled spellF1 is then that count over UNIT,
six decimals written as an integer, and a margin of four millionths is a difference of counts, ten orders of
magnitude above the last bit of a float. The configs are written into a temporary folder (grid_folder), never into
pipeline/benchmark/configs/p27/, whose files the real reports are checked against. No report of a run is read, and no
number here is a figure of any match, but where a docstring quotes R030's.
"""

import contextlib
import io
import json
import math
import os
import re
import tempfile
import tomllib

from pipeline.benchmark import choose, p27_choose, p27_grid, score, sources, touch_stats
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import read
from pipeline.config import load_config
from pipeline.selftest import expect_error, write_text

# Params
PROFILE = "generic"                 # every run of the spec: --profile generic, seeds 0 to 2, the seven DFL matches
MATCHES = sources.MATCH_IDS         # a full run's: its reports carry no .m- suffix, and the bar's p is over seven
UNIT = 1_000_000                    # the possessions of each match of a fake report: a score is a count of millionths
MADE_UP_CONFIG = "# a made-up config: the reports of checks/p27.py are of no run\n"
# The spec's grids, written out here and not read from p27_grid, in tenths of a metre, m/s, frames and hundredths
# (Stage A, 4 x 3 x 4 x 4 = 192), and in events, frames and thousandths (Stage C, 6 x 6 x 4 = 144).
RADII, SPEEDS, MIN_FRAMES, HOLDS = (15, 20, 25, 30), (6, 9, 12), (1, 2, 3, 5), (52, 55, 60, 70)
SWITCH_TEAMS, MIN_POSSESSIONS, WEIGHTS = (1, 2, 3, 4, 6, 8), (1, 10, 15, 25, 35, 50), (100, 200, 300, 500)
# A made-up touch_stats result: a confidence of its own for each (radius, max_rel_speed, min_frames), 0.51 to 0.98.
CONFIDENCE = {(r / 10, float(v), k): round(0.51 + number / 100, 2)
              for number, (r, v, k) in enumerate((r, v, k) for r in RADII for v in SPEEDS for k in MIN_FRAMES)}
# R030's choices, as the chain passed them on: Stage A's touch rule, Stage C's decoder and shape weight.
TOUCH, DECODER, WEIGHT = "1.5,6.0,1,0.6", "6,35", "0.3"
CHOSEN = ("--touch", TOUCH, "--decoder", DECODER, "--weight", WEIGHT)
# Stage D (the spec's): each group in the tie order of the entry that first chose its dimension, as (the stem without
# its prefix, the argument of p27_grid.py that makes it); d_base, the incumbent of every group, changes nothing.
D_GROUPS = {
    "decoder_dead": (("d_sd64_md25", "--switch-dead 64 --min-dead 25"),
                     ("d_sd64_md01", "--switch-dead 64 --min-dead 1"), ("d_base", None),
                     ("d_sd32_md01", "--switch-dead 32 --min-dead 1"),
                     ("d_sd16_md25", "--switch-dead 16 --min-dead 25"),
                     ("d_sd16_md01", "--switch-dead 16 --min-dead 1")),
    "dead_weight": (("d_dw010", "--dead-weight 0.1"), ("d_dw025", "--dead-weight 0.25"),
                    ("d_dw050", "--dead-weight 0.5"), ("d_base", None)),
    "scope": (("d_scope_gaps", "--scope gaps"), ("d_base", None)),
}
D_WRITTEN = ("d_base", "d_sd16_md01", "d_sd16_md25", "d_sd32_md01", "d_sd64_md01", "d_sd64_md25", "d_dw010", "d_dw025",
             "d_dw050", "d_scope_gaps", "d_noshape")     # the eleven of Stage D, in the order p27_grid writes them
# Scores in millionths, (raw, coarse). BASE is R030's Stage C choice (0.5410 and 0.5833, mean 0.56215); LOW is far
# outside the band of anything; a config RAISED is inside the band of BASE with a larger min(raw, coarse).
BASE, LOW, RAISED = (541000, 583300), (400000, 400000), (541010, 583310)
GROUP_LINE = re.compile(r"^group +(\w+) +best (\S+) \(mean [0-9.]+\), tied (\d+), chosen (\S+) \(")


# -- the made-up reports ----------------------------------------------------------------------------------------------

def fake_report(config, matched):
    """A report as run.py writes one for the generic profile's full run, with the fields that choose reads and no other.

    `matched` is the config's pooled spellF1 in millionths, an int that every match then has, or the seven matches' in
    the order of MATCHES. Each match is one half of UNIT true and UNIT predicted possessions, that many of them matched
    at every IoU: its spellF1 is matched / UNIT, and the pooled one the mean of the seven. A match given as (matched,
    possessions) has that many possessions in place of UNIT, and weighs as much in the pooled score. The confusion is
    a perfect one, so that macroF1 says nothing."""
    per_match = [matched] * len(MATCHES) if isinstance(matched, int) else list(matched)
    per_match = [(entry, UNIT) if isinstance(entry, int) else tuple(entry) for entry in per_match]
    assert len(per_match) == len(MATCHES) and all(
        isinstance(count, int) and isinstance(possessions, int) and 0 <= count <= possessions
        for count, possessions in per_match), per_match
    halves = [{"match_id": match_id, "half": 1, "seed": 0, "n_gt": possessions, "n_pred": possessions,
               "tp": [count] * len(score.IOUS), "confusion": [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0]]}
              for match_id, (count, possessions) in zip(MATCHES, per_match)]
    return {"config": {"path": config.replace(os.sep, "/"), "text": MADE_UP_CONFIG},
            "profile": {"name": PROFILE, "version": benchmark_run.PROFILES[PROFILE]}, "seeds": [0, 1, 2],
            "matches": [{"id": match_id} for match_id in MATCHES], "calibration": None, "halves": halves,
            "pooled": {"alive_dead_f1": 0.97}, "possessions_per_graded_minute": {"predicted": 3.0},
            "labels": {"predicted": {"D": 0.45}}}


def reported(root, scores):
    """Write the raw and the coarse report of each stem of `scores`, {stem: (raw, coarse)} as fake_report takes them,
    where p27_choose reads them under the data folder `root`. They all name one made-up config, which is there."""
    config = os.path.join(root, "made_up.toml")
    if not os.path.exists(config):
        write_text(root, "made_up.toml", MADE_UP_CONFIG)
    for stem, pair in scores.items():
        for truth, matched in zip(benchmark_run.TRUTHS, pair):
            path = benchmark_run.report_path(root, PROFILE, stem, truth)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(fake_report(config, matched), handle)


def staged(root, name, stems, **options):
    """p27_choose.stage's choice among `stems` (in that order) on the reports under `root`, its printout swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return p27_choose.stage(name, {stem: None for stem in stems}, root, **options)


def mean_of(choice, stem):
    return choice["rows"][stem]["mean"]["spell"]


def floor_of(choice, stem):
    return min(choice["rows"][stem][truth]["spell"] for truth in benchmark_run.TRUTHS)


def choose_main(root, *argv):
    """(exit code, printed lines) of p27_choose.main(argv) on the reports under `root`."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = p27_choose.main(list(argv) + ["--data-dir", root])
    return code, out.getvalue().splitlines()


def next_call(lines):
    """The arguments of the next p27_grid call as the chain scripts read them from p27_choose's printout: the last
    line that starts with NEXT, without those four characters, trimmed and split at spaces. It must be the last line,
    and the only such line."""
    assert lines and [line for line in lines if line.startswith("NEXT")] == lines[-1:], lines[-3:]
    return lines[-1][4:].split()


def choice_file(root, name):
    return json.loads(read(os.path.join(root, "runs", "p27", name)))


# -- the configs, written where no real report looks ------------------------------------------------------------------

@contextlib.contextmanager
def grid_folder(folder):
    """p27_grid writing into `folder` in place of pipeline/benchmark/configs/p27/, restored afterwards. A check must
    never write there: choose refuses a report whose config has changed, so a made-up config under a real name would
    make the reports of R030 stale."""
    assert not os.path.abspath(folder).startswith(os.path.abspath(p27_grid.ROOT) + os.sep), folder
    real = p27_grid.FOLDER
    p27_grid.FOLDER = folder
    try:
        yield folder
    finally:
        p27_grid.FOLDER = real


def confidence_file(root, name="touch_stats.json", table=None):
    """A touch_stats result file under `root` with `table`'s confidences (default: CONFIDENCE's 48); its path."""
    settings = {}
    for (radius, speed, frames), confidence in (CONFIDENCE if table is None else table).items():
        setting = {"radius": radius, "max_rel_speed": speed, "min_frames": frames}
        settings[touch_stats.label(setting)] = {"setting": setting, "confidence": confidence}
    return write_text(root, name, json.dumps({"profile": PROFILE, "settings": settings}))


def grid_main(folder, *argv):
    """(stems, {stem: text}): the stems that p27_grid.main(argv) printed and the configs it wrote, into `folder`.

    The printout is one line, the stems separated by single spaces, which is how the chain scripts split it. The texts
    are the files' bytes decoded, so a carriage return would show."""
    out = io.StringIO()
    with grid_folder(folder), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = p27_grid.main(list(argv))
    lines = out.getvalue().splitlines()
    assert code == 0 and len(lines) == 1, (code, lines)
    stems = lines[0].split(" ")
    return stems, {stem: read(os.path.join(folder, stem + ".toml"), "rb").decode("utf-8") for stem in stems}


def body(text):
    """A config's lines without its description: what a run of it does."""
    lines = text.split("\n")
    assert sum(line.startswith("description = ") for line in lines) == 1, text
    return [line for line in lines if not line.startswith("description = ")]


def parts(text):
    """(touch, dead, shape or None, decoder, refiners) of a config's text, as TOML reads them."""
    table = tomllib.loads(text)
    sources_by_kind = {entry["kind"]: entry for entry in table["evidence"]}
    assert len(sources_by_kind) == len(table["evidence"]), text
    return (sources_by_kind["controlled_touch"], sources_by_kind["dead_ball_model"],
            sources_by_kind.get("shape_possession"), table["decoder"], table.get("refine", []))


def stage_a_in_tie_order():
    """[(stem, (radius, max_rel_speed, min_frames, hold))] of Stage A in the spec's tie order, in plain loops: the
    larger min_frames, then the smaller radius, the smaller speed, the smaller hold."""
    return [("p27a_r{:02d}_v{:02d}_k{}_h{:02d}".format(r, v, k, h), (r / 10, float(v), k, h / 100))
            for k in sorted(MIN_FRAMES, reverse=True) for r in sorted(RADII) for v in sorted(SPEEDS)
            for h in sorted(HOLDS)]


def stage_c_in_tie_order(prefix="p27"):
    """[(stem, (switch_team, min_possession, weight))] of Stage C in the spec's tie order, in plain loops: the larger
    switch_team, then the larger min_possession, the smaller weight."""
    return [("{}c_t{:02d}_mp{:03d}_w{:03d}".format(prefix, t, mp, w), (t, mp, w / 1000))
            for t in sorted(SWITCH_TEAMS, reverse=True) for mp in sorted(MIN_POSSESSIONS, reverse=True)
            for w in sorted(WEIGHTS)]


def stage_d_reports(root, scores=None, prefix="p27"):
    """Write Stage D's eleven reports under `root`: the incumbent at BASE and each variant at LOW, but for those that
    `scores`, {stem without its prefix: (raw, coarse)}, names."""
    scores = dict(scores or {})
    assert set(scores) <= set(D_WRITTEN), sorted(set(scores) - set(D_WRITTEN))
    reported(root, {prefix + stem: scores.get(stem, BASE if stem == "d_base" else LOW) for stem in D_WRITTEN})


def ran_stage_d(root, scores=None, prefix="p27"):
    """(arguments, {group: (best, tied, chosen)}, lines) of p27_choose.stage_d on stage_d_reports' eleven reports. The
    stems of the groups are given without the prefix."""
    stage_d_reports(root, scores, prefix)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        arguments = p27_choose.stage_d(root, prefix)
    lines = out.getvalue().splitlines()
    groups = {}
    for line in lines:
        found = GROUP_LINE.match(line)
        if found:
            name, best, tied, chosen = found.groups()
            assert best.startswith(prefix) and chosen.startswith(prefix), line
            groups[name] = (best[len(prefix):], int(tied), chosen[len(prefix):])
    assert sorted(groups) == sorted(D_GROUPS), lines
    return arguments, groups, lines


# -- the tie band's edge ----------------------------------------------------------------------------------------------

def test_p27_tie_band_edge_is_compared_on_the_unrounded_scores():
    """R030's two knife-edges, with its own figures, through p27_choose.stage on made-up reports.

    Stage A: the incumbent hold has the best mean (0.558352), h60 is 0.004996 below it and has the larger min(raw,
    coarse), h52 is 0.005054 below: h60 is tied and chosen, h52 is not tied. Stage C: t06_mp025_w300 has the largest
    min of all and is 0.005034 below the best mean: it is not tied, and the choice is t06_mp035_w300. To four decimals,
    as the tables print them, h60 and t06_mp025_w300 are both 0.0050 below their best: only the unrounded scores put
    one inside the band and the other outside. Each is then moved 0.000008 and 0.000038 across the edge, and the
    choice follows: the edge is what decides, and under the raw objective (R008's rule) either moved case would have
    been chosen otherwise."""
    assert p27_choose.BAND == choose.TIE_BAND == 0.005
    a_out, a_best, a_in = ("p27a_r15_v06_k1_h{}".format(hold) for hold in (52, 55, 60))
    c_best, c_chosen, c_out = "p27c_t08_mp050_w300", "p27c_t06_mp035_w300", "p27c_t06_mp025_w300"
    with tempfile.TemporaryDirectory() as root:
        reported(root, {"p27_ref": (538800, 577904), a_out: (527400, 579196), a_best: (538800, 577904),
                        a_in: (546300, 560412)})
        choice = staged(root, "a", [a_out, a_best, a_in])
        assert abs(mean_of(choice, a_best) - 0.558352) < 1e-9
        assert abs(mean_of(choice, a_best) - mean_of(choice, a_in) - 0.004996) < 1e-9, "inside by 0.000004"
        assert abs(mean_of(choice, a_best) - mean_of(choice, a_out) - 0.005054) < 1e-9
        assert floor_of(choice, a_in) > floor_of(choice, a_best) and floor_of(choice, a_out) < floor_of(choice, a_best)
        assert (choice["best"], choice["tied"], choice["chosen"]) == (a_best, [a_best, a_in], a_in), (
            "0.004996 below the best mean is tied, and the larger min(raw, coarse) takes it", choice["tied"])
        assert (choice["objective"], choice["band"], choice["reference"]) == ("mean", 0.005, "p27_ref")
        inside = mean_of(choice, a_best) - mean_of(choice, a_in)

        reported(root, {a_in: (546300, 560396)})                # its mean 0.000008 lower: 0.005004 below the best
        moved = staged(root, "a", [a_out, a_best, a_in])
        assert abs(mean_of(moved, a_best) - mean_of(moved, a_in) - 0.005004) < 1e-9
        assert (moved["tied"], moved["chosen"]) == ([a_best], a_best), (
            "0.005004 below is outside the band, so its larger min(raw, coarse) counts for nothing", moved["tied"])
        raw_rule = choose.choose([a_out, a_best, a_in], "p27_ref", profile=PROFILE, data_dir=root)
        assert raw_rule["chosen"] == a_in != moved["chosen"], "the reports must tell the mean objective from the raw"

        reported(root, {"p27_ref": (538800, 577904), c_best: (524200, 606400), c_chosen: (541000, 583300),
                        c_out: (543700, 576832)})
        choice = staged(root, "c", [c_best, c_chosen, c_out])
        assert abs(mean_of(choice, c_best) - mean_of(choice, c_out) - 0.005034) < 1e-9, "outside by 0.000034"
        assert floor_of(choice, c_out) > floor_of(choice, c_chosen) > floor_of(choice, c_best)
        assert (choice["best"], choice["tied"], choice["chosen"]) == (c_best, [c_best, c_chosen], c_chosen), (
            "0.005034 below the best mean is not tied, although its min(raw, coarse) is the largest", choice["tied"])
        outside = mean_of(choice, c_best) - mean_of(choice, c_out)
        assert "{:.4f}".format(inside) == "{:.4f}".format(outside) == "0.0050" and inside < p27_choose.BAND < outside, (
            "the two edges must read the same when rounded, or a rule on rounded scores would pass this check")

        reported(root, {c_out: (543700, 576908)})               # its mean 0.000038 higher: 0.004996 below the best
        moved = staged(root, "c", [c_best, c_chosen, c_out])
        assert abs(mean_of(moved, c_best) - mean_of(moved, c_out) - 0.004996) < 1e-9
        assert (moved["tied"], moved["chosen"]) == ([c_best, c_chosen, c_out], c_out), moved["tied"]
        raw_rule = choose.choose([c_best, c_chosen, c_out], "p27_ref", profile=PROFILE, data_dir=root)
        assert raw_rule["chosen"] == c_chosen != moved["chosen"]


def test_p27_a_mean_exactly_the_band_below_is_decided_by_the_floats():
    """A mean exactly 0.005 below the best mean, pinned as choose treats it today: it is tied if the floats say so,
    and they do not always say what the paper says.

    The comparison is `mean >= best mean - 0.005` on floats, with no tolerance: a mean that is the edge to the bit is
    tied, and the next float below it is not (_select_mean, on given floats). Through the reports, where a mean is
    (raw + coarse) / 2 of two pooled scores: under a best of 0.70 and 0.50 a config of 0.595 and 0.595 is the edge to
    the bit, so it is tied (and chosen, its min being the larger); under a best of 0.65 and 0.55, which is 0.005 above
    it on paper just the same, the best's mean comes out as 0.6000000000000001 and the edge as 0.5950000000000001, so
    the same config is not tied. An exact tie on paper is decided by the last bit of a sum. No choice of R030 was that
    close (the nearest was 0.000004 inside), and a rule on rounded scores would move the edge by far more; this pins
    what the code does, it does not say that it should."""
    band = p27_choose.BAND
    edge = 0.75 - band
    under = math.nextafter(edge, 0.0)
    raw = {"best": 1.0, "edge": edge, "under": under}           # the best's mean is 0.75 to the bit; the others'
    coarse = {"best": 0.5, "edge": edge, "under": under}        # means are their own scores, (x + x) / 2
    assert choose._select_mean(["best", "edge", "under"], raw, coarse, band) == ("best", ["best", "edge"], "edge")
    assert choose._select_mean(["best", "under"], raw, coarse, band) == ("best", ["best"], "best")
    with tempfile.TemporaryDirectory() as root:
        reported(root, {"p27_ref": BASE, "edge": (595000, 595000), "wide": (700000, 500000),
                        "narrow": (650000, 550000)})
        choice = staged(root, "x", ["wide", "edge"])
        top, mine = mean_of(choice, "wide"), mean_of(choice, "edge")
        assert top == 0.6 and mine == top - band, (repr(top), repr(mine))
        assert (choice["best"], choice["tied"], choice["chosen"]) == ("wide", ["wide", "edge"], "edge"), choice["tied"]
        choice = staged(root, "x", ["narrow", "edge"])
        top, mine = mean_of(choice, "narrow"), mean_of(choice, "edge")
        assert abs(top - mine - band) < 1e-15 and top > 0.6 and mine < top - band, (repr(top), repr(mine))
        assert (choice["best"], choice["tied"], choice["chosen"]) == ("narrow", ["narrow"], "narrow"), choice["tied"]


# -- the grids, their stems and the tie orders ------------------------------------------------------------------------

def test_p27_grids_and_incumbent_are_the_declared_ones():
    """The grids are the spec's (R013's Stage A and Stage C, the spec's Stage D), the incumbent is main.toml as R024
    left it, the rule's constants are the declared ones, and Stage A's touch settings are exactly the 48 that
    touch_stats' r013 grid measures a confidence for."""
    assert (p27_grid.RADII, p27_grid.SPEEDS) == ((1.5, 2.0, 2.5, 3.0), (6.0, 9.0, 12.0))
    assert (p27_grid.MIN_FRAMES, p27_grid.HOLDS) == ((1, 2, 3, 5), (0.52, 0.55, 0.60, 0.70))
    assert (p27_grid.SWITCH_TEAMS, p27_grid.MIN_POSSESSIONS) == ((1, 2, 3, 4, 6, 8), (1, 10, 15, 25, 35, 50))
    assert p27_grid.WEIGHTS == (0.1, 0.2, 0.3, 0.5)
    assert (p27_grid.SWITCH_DEADS, p27_grid.MIN_DEADS) == ((16, 32, 64), (1, 25))
    assert (p27_grid.DEAD_WEIGHTS, p27_grid.SCOPES) == ((0.1, 0.25, 0.5, 1.0), ("gaps", "all"))
    assert (p27_grid.TOUCH, p27_grid.DECODER) == ((1.5, 6.0, 1, 0.55), (4, 35))
    assert (p27_grid.SWITCH_DEAD, p27_grid.MIN_DEAD, p27_grid.DEAD_WEIGHT) == (32, 25, 1.0)
    assert (p27_grid.SHAPE_WEIGHT, p27_grid.SCOPE, p27_grid.MIN_PROB) == (0.2, "all", 0.02)
    assert (p27_choose.PROFILE, p27_choose.REFERENCE, p27_choose.BAND) == ("generic", "p27_ref", 0.005)
    assert p27_choose.BAND == choose.TIE_BAND and p27_choose.PROFILE == benchmark_run.DEFAULT_PROFILE
    measured = {(setting["radius"], setting["max_rel_speed"], setting["min_frames"])
                for setting in touch_stats.R013_GRID}
    assert measured == set(CONFIDENCE) == {setting[:3] for _, setting in stage_a_in_tie_order()}, (
        "Stage A's touch settings must be the ones that touch_stats.py --grid r013 measures")
    assert len(measured) == 48 and len(stage_a_in_tie_order()) == 192 and len(stage_c_in_tie_order()) == 144


def test_p27_stage_a_stems_are_the_grids_in_the_declared_tie_order():
    """p27_choose names Stage A's 192 stems exactly as p27_grid writes them, the same in the same order, and the order
    is the spec's tie order, built here in plain loops: the larger min_frames, then the smaller radius, the smaller
    speed, the smaller hold. The setting that the choice reads from a stem (and prints after NEXT) is the touch rule
    in that stem's config, whose confidence is the touch file's for its radius, speed and frames, and the rest of the
    config is p27_ref's: the incumbent's stem is p27_ref itself."""
    expected = stage_a_in_tie_order()
    stems = [stem for stem, _ in expected]
    assert (stems[0], stems[1], stems[-1]) == ("p27a_r15_v06_k5_h52", "p27a_r15_v06_k5_h55", "p27a_r30_v12_k1_h70")
    assert len(set(stems)) == 192
    table = p27_choose._stage_a()
    assert list(table) == stems, "p27_choose's Stage A is not the declared tie order"
    assert list(table.values()) == [setting for _, setting in expected]
    with tempfile.TemporaryDirectory() as root:
        confidences = confidence_file(root)
        written, texts = grid_main(root, "a", "--confidences", confidences)
        assert written == stems, "p27_grid writes other stems than p27_choose reads, or in another order"
        _, reference = grid_main(root, "ref", "--confidences", confidences)
        for stem, setting in expected:
            touch = parts(texts[stem])[0]
            assert (touch["radius"], touch["max_rel_speed"], touch["min_frames"], touch["hold"]) == setting, stem
            assert touch["confidence"] == CONFIDENCE[setting[:3]], (stem, touch["confidence"])
            assert parts(texts[stem])[1:] == parts(reference["p27_ref"])[1:], stem
        assert body(texts["p27a_r15_v06_k1_h55"]) == body(reference["p27_ref"]), "the incumbent's config is p27_ref"


def test_p27_stage_c_stems_are_the_grids_in_the_declared_tie_order_under_any_prefix():
    """The same for Stage C's 144 stems, under the declared prefix and under another one: the larger switch_team, then
    the larger min_possession, the smaller weight; the decoder and the shape weight that the choice reads from a stem
    are those of its config, and its touch rule is the one Stage A chose, with that rule's confidence."""
    assert [stem for stem, _ in stage_c_in_tie_order()][::143] == ["p27c_t08_mp050_w100", "p27c_t01_mp001_w500"]
    with tempfile.TemporaryDirectory() as root:
        confidences = confidence_file(root)
        for prefix, arguments in (("p27", ()), ("alt55", ("--prefix", "alt55"))):
            expected = stage_c_in_tie_order(prefix)
            stems = [stem for stem, _ in expected]
            table = p27_choose._stage_c(prefix)
            assert list(table) == stems and len(set(stems)) == 144, prefix
            assert list(table.values()) == [setting for _, setting in expected], prefix
            written, texts = grid_main(root, "c", "--confidences", confidences, "--touch", TOUCH, *arguments)
            assert written == stems, prefix
            for stem, (switch_team, min_possession, weight) in expected:
                touch, dead, shape, decoder, refiners = parts(texts[stem])
                assert (decoder["switch_team"], decoder["min_possession"], shape["weight"]) == (
                    switch_team, min_possession, weight), stem
                assert (touch["radius"], touch["max_rel_speed"], touch["min_frames"], touch["hold"],
                        touch["confidence"]) == (1.5, 6.0, 1, 0.6, CONFIDENCE[(1.5, 6.0, 1)]), stem
                assert (decoder["switch_dead"], decoder["min_dead"], dead["weight"], shape["scope"], refiners) == (
                    32, 25, 1.0, "all", []), stem
        assert list(p27_choose._stage_c()) == [stem for stem, _ in stage_c_in_tie_order()], "the default prefix is p27"


def test_p27_an_exact_tie_goes_to_the_stages_tie_order():
    """Of two configs that lead with the same scores, the stage takes the one that the spec's order puts first, level
    by level: Stage A the larger min_frames whatever the rest, then the smaller radius, the smaller speed, the smaller
    hold; Stage C the larger switch_team, then the larger min_possession, the smaller weight. Each pair differs in
    every later dimension the other way, so an order with two levels swapped takes the other config. A pair is given
    to p27_choose.stage as the two entries of the stage's own table (_stage_a(), _stage_c()), in the table's order, on
    made-up reports; the whole grids with every config level are test_p27_choose_main_*'s."""
    with tempfile.TemporaryDirectory() as root:
        reported(root, {"p27_ref": LOW})
        for name, table, pairs in (
                ("a", p27_choose._stage_a(), (
                    ((3.0, 12.0, 5, 0.70), (1.5, 6.0, 3, 0.52)),          # the larger min_frames, whatever the rest
                    ((1.5, 12.0, 1, 0.70), (2.0, 6.0, 1, 0.52)),          # then the smaller radius
                    ((2.0, 6.0, 2, 0.70), (2.0, 9.0, 2, 0.52)),           # then the smaller speed
                    ((2.5, 9.0, 3, 0.55), (2.5, 9.0, 3, 0.60)))),         # then the smaller hold
                ("c", p27_choose._stage_c(), (
                    ((8, 1, 0.5), (6, 50, 0.1)),                          # the larger switch_team, whatever the rest
                    ((4, 35, 0.5), (4, 25, 0.1)),                         # then the larger min_possession
                    ((4, 25, 0.2), (4, 25, 0.3))))):                      # then the smaller weight
            stems = {setting: stem for stem, setting in table.items()}
            assert len(stems) == len(table), "two stems of one setting"
            for first, second in pairs:
                pair = (stems[first], stems[second])
                reported(root, {stem: BASE for stem in pair})
                choice = staged(root, name, [stem for stem in table if stem in pair])
                assert (choice["tied"], choice["chosen"]) == ([stems[first], stems[second]], stems[first]), (
                    name, first, second, choice["tied"], choice["chosen"])


# -- Stage D ----------------------------------------------------------------------------------------------------------

def test_p27_stage_d_reads_the_eleven_configs_the_grid_writes():
    """Stage D is eleven configs, the incumbent and ten variants, and p27_choose reads the reports of exactly the stems
    that p27_grid writes, under the declared prefix and under another: no more (a missing report is refused, by name)
    and no fewer. The incumbent, <prefix>d_base, is Stage C's choice and the reference; each variant is the incumbent
    with its one dimension changed and nothing else."""
    with tempfile.TemporaryDirectory() as root:
        confidences = confidence_file(root)
        for prefix, more in (("p27", ()), ("alt55", ("--prefix", "alt55"))):
            written, texts = grid_main(os.path.join(root, "configs"), "d", "--confidences", confidences, *CHOSEN, *more)
            assert written == [prefix + stem for stem in D_WRITTEN] and len(set(written)) == 11, written
            touch, dead, shape, decoder, refiners = parts(texts[prefix + "d_base"])
            assert (touch["radius"], touch["max_rel_speed"], touch["min_frames"], touch["hold"]) == (1.5, 6.0, 1, 0.6)
            assert decoder == {"kind": "hmm", "switch_team": 6, "switch_dead": 32, "min_possession": 35, "min_dead": 25}
            assert (dead["weight"], shape["weight"], shape["scope"], refiners) == (1.0, 0.3, "all", [])
            changed = {"d_sd16_md01": ("decoder", {"switch_dead": 16, "min_dead": 1}),
                       "d_sd16_md25": ("decoder", {"switch_dead": 16}), "d_sd32_md01": ("decoder", {"min_dead": 1}),
                       "d_sd64_md01": ("decoder", {"switch_dead": 64, "min_dead": 1}),
                       "d_sd64_md25": ("decoder", {"switch_dead": 64}), "d_dw010": ("dead", {"weight": 0.1}),
                       "d_dw025": ("dead", {"weight": 0.25}), "d_dw050": ("dead", {"weight": 0.5}),
                       "d_scope_gaps": ("shape", {"scope": "gaps"}), "d_noshape": ("shape", None)}
            assert sorted(changed) == sorted(D_WRITTEN[1:])
            for stem, (where, values) in changed.items():
                wanted = dict(zip(("touch", "dead", "shape", "decoder", "refiners"), parts(texts[prefix + "d_base"])))
                wanted[where] = None if values is None else dict(wanted[where], **values)
                assert parts(texts[prefix + stem]) == tuple(wanted.values()), (prefix, stem)

            arguments, groups, _ = ran_stage_d(root, prefix=prefix)
            assert arguments == [] and {name: chosen for name, (_, _, chosen) in groups.items()} == {
                name: "d_base" for name in D_GROUPS}, groups
            name = "stage_d_choice.json" if prefix == "p27" else prefix + "_stage_d_choice.json"
            document = choice_file(root, name)
            assert sorted(document["rows"]) == sorted(written), "Stage D must read the reports of the stems written"
            assert document["reference"] == prefix + "d_base"
            for stem in D_WRITTEN:
                for truth in benchmark_run.TRUTHS:
                    path = benchmark_run.report_path(root, PROFILE, prefix + stem, truth)
                    kept = read(path)
                    os.remove(path)
                    expect_error(choose.ChooseError, lambda: stage_d_as_it_is(root, prefix), prefix + stem, "no report")
                    write_text(os.path.dirname(path), os.path.basename(path), kept)
            assert stage_d_as_it_is(root, prefix) == [], "every report is back"


def stage_d_as_it_is(root, prefix):
    """p27_choose.stage_d on the reports that are under `root`, as they are; its printout swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return p27_choose.stage_d(root, prefix)


def test_p27_stage_d_chooses_each_group_by_the_rule_with_the_incumbent_a_candidate():
    """Each group of Stage D is chosen by the rule of Stages A and C (the best mean, the band, the larger min(raw,
    coarse)), the incumbent being one candidate among the others (the advisor's ruling of 9 Nov), not kept unless it
    is beaten. R030's own decoder group first: 64/25 has the best mean, all six are within the band, and the incumbent
    32/25 has the larger min (0.5410 against 0.5407), so nothing changes; without the incumbent in the group the
    choice would be 64/25. Then, one at a time: a variant inside the band with a larger min is taken although the
    incumbent's mean is the best; the same variant 0.005034 below is not; a variant whose mean is better by more than
    the band is taken although its min is lower; one whose mean is better within the band and whose min is lower is
    not."""
    with tempfile.TemporaryDirectory() as root:
        r030 = {"d_sd64_md25": (540700, 583900), "d_sd64_md01": (540600, 583940), "d_sd32_md01": (540500, 583500),
                "d_sd16_md25": (539000, 581200), "d_sd16_md01": (538600, 580800)}
        arguments, groups, lines = ran_stage_d(root, r030)
        assert groups["decoder_dead"] == ("d_sd64_md25", 6, "d_base") and arguments == [], groups
        assert lines[-1] == "CHANGED_GROUPS 0" and not any("CHANGED" in line for line in lines[:-1]), lines
        arguments, groups, _ = ran_stage_d(root, dict(r030, d_base=(540690, 583610)))    # the incumbent's min now lower
        assert groups["decoder_dead"] == ("d_sd64_md25", 6, "d_sd64_md25"), groups
        assert arguments == ["--switch-dead 64 --min-dead 25"], arguments

        for scores, wanted, why in (
                ({"d_dw050": (545000, 575000)}, ("d_base", 2, "d_dw050"),
                 "0.00215 below the incumbent's mean and a larger min: taken, as Stages A and C would"),
                ({"d_dw050": (548000, 566232)}, ("d_base", 1, "d_base"), "0.005034 below: outside the band"),
                ({"d_dw050": (548000, 566308)}, ("d_base", 2, "d_dw050"), "0.004996 below: inside the band"),
                ({"d_dw050": (530000, 610000)}, ("d_dw050", 1, "d_dw050"),
                 "better by more than the band: the incumbent is not tied, so its larger min does not keep it"),
                ({"d_dw050": (540000, 586300)}, ("d_dw050", 2, "d_base"),
                 "the better mean within the band and the lower min: the incumbent stays")):
            arguments, groups, lines = ran_stage_d(root, scores)
            assert groups["dead_weight"] == wanted, (why, groups["dead_weight"])
            changed = wanted[2] != "d_base"
            assert arguments == (["--dead-weight 0.5"] if changed else []), (why, arguments)
            assert lines[-1] == "CHANGED_GROUPS {}".format(int(changed)), (why, lines[-1])
            assert groups["decoder_dead"][2] == groups["scope"][2] == "d_base", "the other groups are not touched"


def test_p27_stage_d_tie_orders_are_those_of_the_entries_that_chose_them():
    """An exact tie inside a group of Stage D goes to the order of the entry that first chose the dimension: the
    larger switch_dead, then the larger min_dead (R010); the smaller dead-ball weight (R009); `gaps` before `all`
    (R011). With a whole group level with the incumbent the first of its order is taken, and of two neighbours of the
    order that lead the group with the same scores, the one named first; the incumbent has its place in each order."""
    assert [stem for stem, _ in D_GROUPS["decoder_dead"]] == [
        "d_base" if (sd, md) == (32, 25) else "d_sd{:02d}_md{:02d}".format(sd, md)
        for sd in (64, 32, 16) for md in (25, 1)]
    assert [stem for stem, _ in D_GROUPS["dead_weight"]] == ["d_dw010", "d_dw025", "d_dw050", "d_base"]
    assert [stem for stem, _ in D_GROUPS["scope"]] == ["d_scope_gaps", "d_base"]
    with tempfile.TemporaryDirectory() as root:
        for name, order in D_GROUPS.items():
            stems = [stem for stem, _ in order]
            arguments = dict(order)
            found, groups, _ = ran_stage_d(root, {stem: BASE for stem in stems})
            assert groups[name] == (stems[0], len(stems), stems[0]), (name, groups[name])
            assert found == [arguments[stems[0]]], (name, found)
            for first, second in list(zip(stems, stems[1:]))[1:]:       # the first pair is the level group's own case
                scores = {stem: BASE for stem in stems}
                scores.update({first: RAISED, second: RAISED})
                found, groups, _ = ran_stage_d(root, scores)
                assert groups[name] == (first, len(stems), first), (name, first, second, groups[name])
                assert found == ([arguments[first]] if arguments[first] else []), (name, first, second, found)


def test_p27_stage_d_keeps_the_shape_source_unless_the_config_without_it_is_better():
    """The shape source stays if the config with it is not worse than the config without it, on the mean of the two
    truths: it stays at the same scores (the same mean to the bit) and at a lower mean, by 0.000001 or with the larger
    min(raw, coarse) on the side of the config without it (this is not the band rule), and --no-shape is returned,
    and counted as a changed group, as soon as the config without it has the better mean, by 0.000001 here, whatever
    its min."""
    with tempfile.TemporaryDirectory() as root:
        for without, stays in ((BASE, True), ((540999, 583299), True), ((560000, 564000), True), (LOW, True),
                               ((541001, 583301), False), ((520000, 604700), False)):
            arguments, _, lines = ran_stage_d(root, {"d_noshape": without})
            shape = [line for line in lines if line.startswith("group     shape ")]
            assert len(shape) == 1 and shape[0].endswith(
                "the shape source stays" if stays else "the shape source is DROPPED"), (without, shape)
            assert arguments == ([] if stays else ["--no-shape"]), (without, arguments)
            assert lines[-1] == "CHANGED_GROUPS {}".format(0 if stays else 1), (without, lines[-1])


def test_p27_stage_d_returns_the_changes_and_counts_the_groups():
    """What Stage D hands on: no argument and CHANGED_GROUPS 0 when every group keeps the incumbent, the one argument
    and 1 for one group, both arguments and 2 for two (the chain stops above 1: the spec has the combined config run),
    in the order decoder, dead-ball weight, scope, shape. p27_choose.py d prints them after NEXT on its last line, the
    count on the line before, and exits 0 each time. And an argument means its variant: p27_grid.py final, given the
    argument that Stage D returns for a variant, writes that variant's own config, the description aside."""
    with tempfile.TemporaryDirectory() as root:
        winner = (560000, 600000)                               # better than BASE by more than the band, under both
        for scores, wanted in (
                ({}, []),
                ({"d_dw050": winner}, ["--dead-weight 0.5"]),
                ({"d_sd64_md01": winner, "d_noshape": winner}, ["--switch-dead 64 --min-dead 1", "--no-shape"]),
                ({"d_scope_gaps": winner, "d_dw025": winner}, ["--dead-weight 0.25", "--scope gaps"]),
                ({"d_noshape": winner, "d_scope_gaps": winner, "d_dw010": winner, "d_sd16_md25": winner},
                 ["--switch-dead 16 --min-dead 25", "--dead-weight 0.1", "--scope gaps", "--no-shape"])):
            stage_d_reports(root, scores)
            code, lines = choose_main(root, "d")
            assert code == 0 and lines[-2:] == ["CHANGED_GROUPS {}".format(len(wanted)),
                                                "NEXT " + " ".join(wanted)], (wanted, lines[-2:])
            assert next_call(lines) == " ".join(wanted).split()
            assert sum(line.endswith(": CHANGED") for line in lines) == len(wanted) - ("--no-shape" in wanted), lines

        confidences = confidence_file(root)
        folder = os.path.join(root, "configs")
        _, variants = grid_main(folder, "d", "--confidences", confidences, *CHOSEN)
        for stem in D_WRITTEN[1:]:
            arguments, _, _ = ran_stage_d(root, {stem: winner})
            assert len(arguments) == 1, (stem, arguments)
            _, final = grid_main(folder, "final", "--name", "x", "--confidences", confidences, *CHOSEN,
                                 *arguments[0].split())
            assert body(final["x"]) == body(variants["p27" + stem]), (stem, arguments)
        _, final = grid_main(folder, "final", "--name", "x", "--confidences", confidences, *CHOSEN)
        assert body(final["x"]) == body(variants["p27d_base"]), "no argument is the incumbent"


# -- the guard and the bar --------------------------------------------------------------------------------------------

def test_p27_guard_keeps_a_shift_only_above_zero_and_inside_the_band():
    """The shift's guard: the shifted config is kept only if its mean is above the unshifted one's, strictly, and
    neither truth falls by more than 0.005. R030's own case is kept: mean +0.000029, raw +0.0004 and coarse -0.0004.
    Not kept: the same scores (a delta of exactly 0), the two truths moving 0.001 in opposite directions (0 to the
    bit), a mean 0.000029 lower, and a mean that is higher while one truth, either of them, is 0.005034 lower; that
    truth 0.004996 lower is still inside the band, and kept, and so is a truth that is lower by the band itself, to
    the bit (the comparison is `>=`). p27_choose.py guard exits 0 for kept and 2 for not, and 1 without its two stems
    or their reports."""
    cases = (((541429, 582929), True, "R030's: mean +0.000029, raw +0.000429, coarse -0.000371"),
             ((541000, 583300), False, "the same scores: the mean is not above"),
             ((542000, 582300), False, "raw +0.001 and coarse -0.001: a mean delta of 0 to the bit"),
             ((540571, 583671), False, "mean -0.000029"),
             ((547000, 578266), False, "mean +0.000483, but coarse 0.005034 lower"),
             ((535966, 590000), False, "mean +0.000833, but raw 0.005034 lower"),
             ((547000, 578304), True, "mean +0.000502, and coarse 0.004996 lower: inside the band"),
             ((536004, 590000), True, "mean +0.000852, and raw 0.004996 lower: inside the band"),
             ((541001, 583300), True, "raw +0.000001 alone: above is enough, nothing asks for a significant gain"))
    with tempfile.TemporaryDirectory() as root:
        reported(root, {"base": BASE})
        for shifted, kept, why in cases:
            reported(root, {"shifted": shifted})
            code, lines = choose_main(root, "guard", "shifted", "base")
            assert code == (0 if kept else 2), (why, code, lines)
            assert len(lines) == 1 and lines[0].startswith("guard     shifted against base: mean "), lines
            assert lines[0].endswith(": kept" if kept else ": NOT kept"), (why, lines[0])
            with contextlib.redirect_stdout(io.StringIO()):
                assert p27_choose.guard("shifted", "base", root) is kept, why
        reported(root, {"shifted": (542000, 582300)})
        versus = choose.choose(["shifted"], "base", profile=PROFILE, objective="mean", data_dir=root)["versus"]
        assert versus["mean"]["pooled_delta"] == 0.0 and versus["raw"]["pooled_delta"] > 0.0009, (
            "the opposite moves must be a mean delta of exactly 0, or the case pins nothing",
            repr(versus["mean"]["pooled_delta"]))
        # A fall of the band itself, to the bit: 0.010000 to 0.005000 is one, the difference of those two floats
        # being the float of 0.005.
        reported(root, {"low": (10000, 500000), "lower": (5000, 506000)})
        versus = choose.choose(["lower"], "low", profile=PROFILE, objective="mean", data_dir=root)["versus"]
        assert versus["raw"]["pooled_delta"] == -p27_choose.BAND and versus["mean"]["pooled_delta"] > 0, (
            "the fall must be the band to the bit, or it pins nothing", repr(versus["raw"]["pooled_delta"]))
        code, lines = choose_main(root, "guard", "lower", "low")
        assert code == 0 and lines[0].endswith(": kept"), (code, lines)
        code, lines = choose_main(root, "guard", "shifted")
        assert code == 1 and len(lines) == 1 and lines[0].startswith("FAILED "), (code, lines)
        code, lines = choose_main(root, "guard", "shifted", "absent")
        assert code == 1 and lines[-1].startswith("FAILED ") and "absent" in lines[-1], (code, lines)


def test_p27_bar_needs_both_yardsticks_beaten_with_p_at_most_005():
    """The final bar (R010's): the config beats nearest_w50 under the raw truth and nearest_w125 under the coarse
    truth, each with a sign-flip p of at most 0.05 over the seven matches, and nothing else is asked. Seven wins in
    seven under each pass (p 0.0156, R030's case, with its figures). It fails when either comparison is lost, although
    the lost one's p is 0.0156 too; it passes whatever the other two pairings say (nearest_w50 under the coarse truth
    and nearest_w125 under the raw one are made unbeatable here); six wins and a loss with p 0.0469 pass, and five
    wins and two losses with p 0.0625 fail although the pooled delta is above 0. Beating is strict: a config level
    with the yardstick on the pooled score fails, with p 1.0 when every match is level and with p 0.0312 when six
    matches win and a large seventh takes it all back. p27_choose.py bar exits 0 for a pass, 2 for a fail and 1
    without its stem or its reports."""
    w50, w125 = 188500, 209200                                  # R030's yardsticks on generic, under their own truth
    seven = len(MATCHES)

    def beside(yardstick, deltas):
        return [yardstick + 1000 * delta for delta in deltas]

    cases = (((541400, 582900), True, ("p 0.0156, wins 7 of 7: pass",) * 2, "R030's final config"),
             ((541400, 200000), False, ("wins 7 of 7: pass", "p 0.0156, wins 0 of 7: FAIL"), "coarse lost"),
             ((180000, 582900), False, ("p 0.0156, wins 0 of 7: FAIL", "wins 7 of 7: pass"), "raw lost"),
             ((beside(w50, (1, -2, 3, 4, 5, 6, 7)), beside(w125, (1, -2, 3, 4, 5, 6, 7))), True,
              ("p 0.0469, wins 6 of 7: pass",) * 2, "six wins and a small loss"),
             ((beside(w50, (-1, -2, 4, 5, 6, 7, 8)), 582900), False,
              ("delta +0.0039, p 0.0625, wins 5 of 7: FAIL", "wins 7 of 7: pass"), "raw above, p over 0.05"),
             ((541400, beside(w125, (-1, -2, 4, 5, 6, 7, 8))), False,
              ("wins 7 of 7: pass", "delta +0.0039, p 0.0625, wins 5 of 7: FAIL"), "coarse above, p over 0.05"),
             ((w50, 582900), False, ("delta +0.0000, p 1.0000, wins 0 of 7: FAIL", "pass"), "level with nearest_w50"))
    with tempfile.TemporaryDirectory() as root:
        reported(root, {"nearest_w50": (w50, 999000), "nearest_w125": (999000, w125)})
        for scores, passes, endings, why in cases:
            reported(root, {"final": scores})
            code, lines = choose_main(root, "bar", "final")
            assert code == (0 if passes else 2), (why, code, lines)
            assert len(lines) == 2 and lines[0].startswith("bar       final against nearest_w50 under the raw truth: ")
            assert lines[1].startswith("bar       final against nearest_w125 under the coarse truth: "), lines[1]
            assert all(line.endswith(ending) for line, ending in zip(lines, endings)), (why, lines)
            assert all("of {}: ".format(seven) in line for line in lines), lines
            with contextlib.redirect_stdout(io.StringIO()):
                assert p27_choose.bar("final", root) is passes, why
        # Level with nearest_w50 on the pooled score, to the bit, while six matches of the seven win (p 0.0312): the
        # seventh holds sixty times the possessions of each of the others and loses as many as the six gain.
        heavy = 60
        reported(root, {"nearest_w50": ([w50] * 6 + [(heavy * w50, heavy * UNIT)], 999000),
                        "final": ([w50 + 1000] * 6 + [(heavy * w50 - 6000, heavy * UNIT)], 582900)})
        versus = choose.choose(["final"], "nearest_w50", profile=PROFILE, objective="mean", data_dir=root)["versus"]
        assert versus["raw"]["pooled_delta"] == 0.0 and versus["raw"]["p_sign_flip"] == 4 / 128, versus["raw"]
        code, lines = choose_main(root, "bar", "final")
        assert code == 2 and lines[0].endswith("delta +0.0000, p 0.0312, wins 6 of 7: FAIL"), (code, lines)
        assert lines[1].endswith("wins 7 of 7: pass"), lines[1]
        code, lines = choose_main(root, "bar")
        assert code == 1 and len(lines) == 1 and lines[0].startswith("FAILED "), (code, lines)
        code, lines = choose_main(root, "bar", "absent")
        assert code == 1 and lines[-1].startswith("FAILED ") and "absent" in lines[-1], (code, lines)


# -- p27_grid's configs -----------------------------------------------------------------------------------------------

def test_p27_grid_configs_load_as_the_runner_loads_them():
    """Every config that p27_grid writes (the reference, Stage A's 192, Stage C's 144, Stage D's 11 and the final one,
    with and without the shift and the shape source) loads through pipeline.config.load_config against the real
    components' signatures and passes what the benchmark runner checks before it loads a match (decoded per half, a
    name without a dot, no name twice). Each is the touch rule, the dead-ball model and the shape model into the hmm,
    in that order, with the models' paths and min_prob 0.02; the files are LF only and end with one newline."""
    with tempfile.TemporaryDirectory() as root:
        confidences = ("--confidences", confidence_file(root))
        for number, arguments in enumerate((
                ("ref",), ("a",), ("c", "--touch", TOUCH), ("d", *CHOSEN), ("final", *CHOSEN),
                ("final", *CHOSEN, "--shift", "1,-2,0", "--name", "shifted"),
                ("final", *CHOSEN, "--no-shape", "--scope", "gaps", "--switch-dead", "64", "--min-dead", "1",
                 "--dead-weight", "0.25", "--name", "bare"))):
            folder = os.path.join(root, "stage{}".format(number))
            stems, texts = grid_main(folder, *arguments, *confidences)
            assert sorted(os.listdir(folder)) == sorted(stem + ".toml" for stem in stems), arguments
            paths = [os.path.join(folder, stem + ".toml") for stem in stems]
            loaded, dropped = benchmark_run._load_configs(paths)
            assert [config.stem for config in loaded] == stems and not any(dropped.values()), arguments
            for stem, config in zip(stems, loaded):
                text = texts[stem]
                assert config.text == text and "\r" not in text and text.endswith("\n") and not text.endswith("\n\n")
                assert config.scope == "halves" and config.seed == 0 and "." not in stem, stem
                kinds = [component.kind for component in config.evidence]
                assert kinds == ["controlled_touch", "dead_ball_model", "shape_possession"][:len(kinds)], (stem, kinds)
                assert len(kinds) == (2 if stem in ("p27d_noshape", "bare") else 3), (stem, kinds)
                assert config.evidence[1].params == {"model": "pipeline/models/dead_ball_hgb.json"}, stem
                if len(kinds) == 3:
                    assert {key: value for key, value in config.evidence[2].params.items() if key != "scope"} == {
                        "model": "pipeline/models/shape_possession_hgb.json", "min_prob": 0.02}, stem
                assert config.decoder.kind == "hmm" and sorted(config.decoder.params) == [
                    "min_dead", "min_possession", "switch_dead", "switch_team"], stem
                assert [component.kind for component in config.refine] == (
                    ["shift_edges"] if stem == "shifted" else []), stem
                assert config.description and config.description == tomllib.loads(text)["description"], stem
        assert load_config(os.path.join(root, "stage0", "p27_ref.toml")).stem == "p27_ref"


def test_p27_grid_final_writes_the_shift_only_when_given():
    """`final --shift a,b,c` adds exactly one shift_edges refiner with team a, into_dead b and out_of_dead c, zeros
    and negative values included, and says nothing else differently; without --shift there is no refiner and the
    description says pre-shift. The stem is --name's, p27_final when it is not given. A --shift that is not three
    values is refused. One whose first value is negative is, today, refused by the argument parser (it reads -1,2,0 as
    an option); what is held here is only that it is never written as another shift: either those three values or a
    refusal that writes nothing."""
    with tempfile.TemporaryDirectory() as root:
        common = ("final", "--confidences", confidence_file(root), *CHOSEN)
        stems, plain = grid_main(root, *common)
        assert stems == ["p27_final"] and parts(plain["p27_final"])[4] == []
        assert tomllib.loads(plain["p27_final"])["description"] == "P27: the chosen pipeline, pre-shift"
        stems, named = grid_main(root, *common, "--name", "p27_preshift")
        assert stems == ["p27_preshift"] and named["p27_preshift"] == plain["p27_final"]
        for shift, values in (("1,-2,0", (1, -2, 0)), ("0,0,0", (0, 0, 0)), ("3,4,-5", (3, 4, -5))):
            stems, shifted = grid_main(root, *common, "--shift", shift)
            assert stems == ["p27_final"]
            text = shifted["p27_final"]
            assert parts(text)[4] == [{"kind": "shift_edges", "team": values[0], "into_dead": values[1],
                                       "out_of_dead": values[2]}], shift
            assert parts(text)[:4] == parts(plain["p27_final"])[:4], shift
            block = ["", "[[refine]]", 'kind = "shift_edges"', "team = {}".format(values[0]),
                     "into_dead = {}".format(values[1]), "out_of_dead = {}".format(values[2]), ""]
            assert body(text) == body(plain["p27_final"])[:-1] + block, shift
            assert tomllib.loads(text)["description"] == "P27: the chosen pipeline", shift
        for shift in ("1,2", "1,2,3,4"):
            expect_error(SystemExit, lambda: grid_main(root, *common, "--shift", shift, "--name", "bad"),
                         "expected 3", shift)
        expect_error(ValueError, lambda: grid_main(root, *common, "--shift", "1,2,x", "--name", "bad"), "'x'")
        assert not os.path.exists(os.path.join(root, "bad.toml"))
        try:
            _, negative = grid_main(root, *common, "--shift", "-1,2,0", "--name", "negative")
        except SystemExit as stop:
            assert stop.code not in (0, None) and not os.path.exists(os.path.join(root, "negative.toml")), stop.code
        else:
            assert parts(negative["negative"])[4] == [
                {"kind": "shift_edges", "team": -1, "into_dead": 2, "out_of_dead": 0}], negative["negative"]


def test_p27_grid_no_shape_drops_the_shape_source_and_nothing_else():
    """`final --no-shape` writes the config without its shape_possession entry, and every other line as it was; the
    other arguments of Stage D each change their own value and nothing else, and the defaults are the incumbent's."""
    with tempfile.TemporaryDirectory() as root:
        common = ("final", "--confidences", confidence_file(root), *CHOSEN)
        _, texts = grid_main(root, *common)
        full = texts["p27_final"]
        touch, dead, shape, decoder, refiners = parts(full)
        assert (dead["weight"], shape["weight"], shape["scope"], decoder["switch_dead"], decoder["min_dead"]) == (
            1.0, 0.3, "all", 32, 25)
        _, texts = grid_main(root, *common, "--no-shape")
        assert parts(texts["p27_final"]) == (touch, dead, None, decoder, refiners)
        block = ["", "[[evidence]]", 'kind = "shape_possession"', "weight = 0.3",
                 'model = "pipeline/models/shape_possession_hgb.json"', 'scope = "all"', "min_prob = 0.02"]
        lines = body(full)
        start = lines.index('kind = "shape_possession"') - 2
        assert lines[start:start + len(block)] == block and body(texts["p27_final"]) == (
            lines[:start] + lines[start + len(block):])
        for arguments, wanted in (
                (("--switch-dead", "64"), (touch, dead, shape, dict(decoder, switch_dead=64), refiners)),
                (("--min-dead", "1"), (touch, dead, shape, dict(decoder, min_dead=1), refiners)),
                (("--dead-weight", "0.25"), (touch, dict(dead, weight=0.25), shape, decoder, refiners)),
                (("--scope", "gaps"), (touch, dead, dict(shape, scope="gaps"), decoder, refiners))):
            _, texts = grid_main(root, *common, *arguments)
            assert parts(texts["p27_final"]) == wanted, arguments


def test_p27_grid_confidence_is_the_touch_files_and_a_missing_setting_is_refused():
    """The confidence that p27_grid writes is the one the touch_stats file holds for the rule's radius, relative speed
    and min_frames, whatever the hold, and it changes with the file: it is read, never typed in (Stage D and the final
    config here; Stage A's and Stage C's are held to the file by the checks of their stems). A setting that the file
    does not hold is refused, by its three values, and so is a file with a setting that has no confidence; a --touch
    that is not four values is refused too. Such a refusal writes no config."""
    with tempfile.TemporaryDirectory() as root:
        first = confidence_file(root)
        other = confidence_file(root, "other.json", {key: round(1.49 - value, 2) for key, value in CONFIDENCE.items()})
        assert len(set(CONFIDENCE.values())) == 48 and CONFIDENCE[(1.5, 6.0, 1)] == 0.51
        folder = os.path.join(root, "configs")
        rest = ("--decoder", DECODER, "--weight", WEIGHT)
        for touch, key, stages in (("1.5,6.0,1,0.6", (1.5, 6.0, 1), (("d", *rest), ("final", *rest))),
                                   ("1.5,6.0,1,0.52", (1.5, 6.0, 1), (("final", *rest),)),
                                   ("3.0,12.0,5,0.7", (3.0, 12.0, 5), (("d", *rest), ("final", *rest))),
                                   ("2,9,3,0.55", (2.0, 9.0, 3), (("final", *rest),)),
                                   ("2.5,6.0,2,0.55", (2.5, 6.0, 2), (("d", *rest), ("final", *rest)))):
            for stage in stages:
                for path, wanted in ((first, CONFIDENCE[key]), (other, round(1.49 - CONFIDENCE[key], 2))):
                    _, texts = grid_main(folder, *stage, "--confidences", path, "--touch", touch)
                    found = {parts(text)[0]["confidence"] for text in texts.values()}
                    assert found == {wanted}, (touch, stage, found, wanted)
        for path, wanted in ((first, 0.51), (other, 0.98)):     # the incumbent, 1.5 m / 6.0 m/s / 1 frame
            _, texts = grid_main(folder, "ref", "--confidences", path)
            assert parts(texts["p27_ref"])[0]["confidence"] == wanted

        empty = os.path.join(root, "empty")
        os.makedirs(empty)
        for touch, named in (("1.0,6.0,1,0.6", "radius 1.0 m"), ("1.5,7.0,1,0.6", "speed 7.0 m/s"),
                             ("1.5,6.0,4,0.6", "min_frames 4")):
            for stage in (("c",), ("d", *rest), ("final", *rest)):
                expect_error(SystemExit, lambda: grid_main(empty, *stage, "--confidences", first, "--touch", touch),
                             "no measured confidence", named)
        unmeasured = confidence_file(root, "unmeasured.json", {**CONFIDENCE, (2.0, 9.0, 3): None})
        for stage in (("ref",), ("a",), ("final", *CHOSEN)):
            expect_error(SystemExit, lambda: grid_main(empty, *stage, "--confidences", unmeasured), "no confidence",
                         "unmeasured.json")
        expect_error(SystemExit, lambda: grid_main(empty, "c", "--confidences", first, "--touch", "1.5,6.0,1"),
                     "expected 4", "1.5,6.0,1")
        assert os.listdir(empty) == [], "a refusal writes no config"
        # Stage A needs each of its 48 settings: with one missing from the file it is refused, by that setting.
        short = confidence_file(root, "short.json", {key: value for key, value in CONFIDENCE.items()
                                                     if key != (2.0, 9.0, 3)})
        partial = os.path.join(root, "partial")
        os.makedirs(partial)
        expect_error(SystemExit, lambda: grid_main(partial, "a", "--confidences", short),
                     "no measured confidence", "radius 2.0 m", "speed 9.0 m/s", "min_frames 3")
        assert not any(name.startswith("p27a_r20_v09_k3_") for name in os.listdir(partial)), os.listdir(partial)


def test_p27_grid_prefix_renames_stages_c_and_d_and_changes_no_config():
    """--prefix changes the stems of Stages C and D, p27 becoming the prefix, in the same order, and nothing in any
    config: each has the text of the declared branch's config of the same place. The reference, Stage A and the final
    config have no prefix: with one given they are written under the declared names, with the same text."""
    with tempfile.TemporaryDirectory() as root:
        confidences = ("--confidences", confidence_file(root))
        for stage in (("c", "--touch", TOUCH), ("d", *CHOSEN)):
            declared, texts = grid_main(os.path.join(root, "declared"), *stage, *confidences)
            renamed, others = grid_main(os.path.join(root, "alt55"), *stage, *confidences, "--prefix", "alt55")
            assert all(stem.startswith("p27" + stage[0] + "_") for stem in declared), stage[0]
            assert renamed == ["alt55" + stem[len("p27"):] for stem in declared], stage[0]
            assert [others[stem] for stem in renamed] == [texts[stem] for stem in declared], stage[0]
        for stage in (("ref",), ("final", *CHOSEN, "--shift", "1,-2,0")):
            declared, texts = grid_main(os.path.join(root, "declared"), *stage, *confidences)
            renamed, others = grid_main(os.path.join(root, "alt55"), *stage, *confidences, "--prefix", "alt55")
            assert renamed == declared and others == texts, stage[0]
        renamed, _ = grid_main(os.path.join(root, "alt55"), "a", *confidences, "--prefix", "alt55")
        assert renamed == [stem for stem, _ in stage_a_in_tie_order()], "Stage A has no prefix"


# -- p27_choose.py's exit codes and its NEXT line ---------------------------------------------------------------------

def test_p27_choose_main_stage_a_names_r030s_touch_rule_for_the_next_call():
    """p27_choose.py a on Stage A's whole grid. With R030's figures (the incumbent hold has the best mean, four configs
    are within 0.005 of it, and h60, 0.004996 below, has the larger min(raw, coarse)) it exits 0, its last line is
    `NEXT --touch 1.5,6.0,1,0.6` and the selection it writes says the same. The line's arguments are TOUCH's, which
    the checks of p27_grid above pass to it as they are, as the chain does. With all 192 configs level, the line names
    the first of the tie order, 1.5,6.0,5,0.52."""
    scores = {"p27_ref": (538800, 577904), "p27a_r15_v06_k1_h55": (538800, 577904),
              "p27a_r15_v06_k2_h55": (533400, 577700), "p27a_r20_v06_k1_h55": (535600, 575200),
              "p27a_r15_v06_k1_h60": (546300, 560412), "p27a_r15_v06_k1_h52": (527400, 579196)}
    stems = [stem for stem, _ in stage_a_in_tie_order()]
    with tempfile.TemporaryDirectory() as root:
        reported(root, {stem: scores.get(stem, LOW) for stem in ["p27_ref"] + stems})
        code, lines = choose_main(root, "a")
        assert code == 0 and lines[-1] == "NEXT --touch 1.5,6.0,1,0.6", (code, lines[-1])
        document = choice_file(root, "stage_a_choice.json")
        assert (document["best"], document["chosen"]) == ("p27a_r15_v06_k1_h55", "p27a_r15_v06_k1_h60")
        assert document["tied"] == ["p27a_r15_v06_k2_h55", "p27a_r15_v06_k1_h55", "p27a_r15_v06_k1_h60",
                                    "p27a_r20_v06_k1_h55"], document["tied"]
        assert document["order"] == stems and document["reference"] == "p27_ref"
        assert (document["objective"], document["band"], document["profile"]["name"]) == ("mean", 0.005, "generic")
        assert "chosen    p27a_r15_v06_k1_h60" in lines, "the printout names the choice"
        assert next_call(lines) == ["--touch", TOUCH], "what the chain hands to p27_grid.py c, d and final"

        reported(root, {stem: LOW for stem in scores})
        code, lines = choose_main(root, "a")
        assert code == 0 and lines[-1] == "NEXT --touch 1.5,6.0,5,0.52", (code, lines[-1])
        document = choice_file(root, "stage_a_choice.json")
        assert (document["best"], document["tied"], document["chosen"]) == (stems[0], stems, stems[0])


def test_p27_choose_main_stage_c_names_r030s_decoder_and_notes_an_edge_of_the_grid():
    """p27_choose.py c on Stage C's whole grid. With R030's figures (t08_mp050_w300 has the best mean, t06_mp035_w300
    the larger min(raw, coarse) of the five within 0.005 of it here, and t06_mp025_w300, with a larger min still, is
    0.005034 below) it exits 0 and its last line is `NEXT --decoder 6,35 --weight 0.3`, with no note, since that choice
    is on no edge of the grid; the line's arguments are DECODER's and WEIGHT's, which the checks of p27_grid above
    pass to it beside Stage A's, as the chain does. With all 144 configs level, the line names the first of the tie
    order, 8,50 and 0.1, and a note on the line before it names the three dimensions, each at its first or its last
    value. With --prefix the stems read are the prefix's, against the same p27_ref, the selection is written under the
    prefix's name beside the declared branch's, and a choice at min_possession 1 is noted for that dimension alone."""
    scores = {"p27_ref": (538800, 577904), "p27c_t08_mp050_w300": (524200, 606400),
              "p27c_t06_mp050_w300": (531200, 598100), "p27c_t08_mp035_w300": (533500, 593800),
              "p27c_t08_mp025_w300": (538200, 589000), "p27c_t06_mp035_w300": (541000, 583300),
              "p27c_t06_mp025_w300": (543700, 576832)}
    stems = [stem for stem, _ in stage_c_in_tie_order()]
    note = "note      the choice is on the edge of the grid in: {} (reported, not extended)"
    with tempfile.TemporaryDirectory() as root:
        reported(root, {stem: scores.get(stem, LOW) for stem in ["p27_ref"] + stems})
        code, lines = choose_main(root, "c")
        assert code == 0 and lines[-1] == "NEXT --decoder 6,35 --weight 0.3", (code, lines[-1])
        assert not any(line.startswith("note") for line in lines), "6, 35 and 0.3 are on no edge of the grid"
        document = choice_file(root, "stage_c_choice.json")
        assert (document["best"], document["chosen"]) == ("p27c_t08_mp050_w300", "p27c_t06_mp035_w300")
        assert document["tied"] == ["p27c_t08_mp050_w300", "p27c_t08_mp035_w300", "p27c_t08_mp025_w300",
                                    "p27c_t06_mp050_w300", "p27c_t06_mp035_w300"], document["tied"]
        assert document["order"] == stems and document["reference"] == "p27_ref"
        assert next_call(lines) == ["--decoder", DECODER, "--weight", WEIGHT], "what the chain hands to p27_grid.py d"

        reported(root, {stem: LOW for stem in scores})
        code, lines = choose_main(root, "c")
        assert code == 0 and lines[-2:] == [note.format("switch_team, min_possession, weight"),
                                            "NEXT --decoder 8,50 --weight 0.1"], (code, lines[-2:])
        assert next_call(lines) == ["--decoder", "8,50", "--weight", "0.1"]
        level = choice_file(root, "stage_c_choice.json")
        assert (level["best"], level["tied"], level["chosen"]) == (stems[0], stems, stems[0])

        others = [stem for stem, _ in stage_c_in_tie_order("alt55")]
        reported(root, {stem: (600000, 650000) if stem == "alt55c_t06_mp001_w300" else LOW for stem in others})
        code, lines = choose_main(root, "c", "--prefix", "alt55")
        assert code == 0 and lines[-2:] == [note.format("min_possession"),
                                            "NEXT --decoder 6,1 --weight 0.3"], (code, lines[-2:])
        document = choice_file(root, "alt55_stage_c_choice.json")
        assert document["chosen"] == "alt55c_t06_mp001_w300" and document["reference"] == "p27_ref"
        assert document["order"] == others
        assert choice_file(root, "stage_c_choice.json") == level, "the declared branch's selection is not touched"


def test_p27_choose_main_fails_without_a_next_line_when_a_report_is_missing():
    """A stage whose reports are not all there exits 1 with one FAILED line that names the first missing one, prints
    no NEXT line for a chain to pick up and writes no selection: Stage A, Stage C under either prefix and Stage D
    under either, with no report at all and with the reference's alone."""
    first = {("a",): "p27a_r15_v06_k5_h52", ("c",): "p27c_t08_mp050_w100", ("c", "--prefix", "alt55"):
             "alt55c_t08_mp050_w100", ("d",): "p27d_sd64_md25", ("d", "--prefix", "alt55"): "alt55d_sd64_md25"}
    with tempfile.TemporaryDirectory() as root:
        for present, missing in (({}, {("a",): "p27_ref", ("c",): "p27_ref", ("c", "--prefix", "alt55"): "p27_ref",
                                       ("d",): "p27d_base", ("d", "--prefix", "alt55"): "alt55d_base"}),
                                 ({"p27_ref": BASE, "p27d_base": BASE, "alt55d_base": BASE}, first)):
            reported(root, present)
            for arguments, stem in missing.items():
                code, lines = choose_main(root, *arguments)
                assert code == 1 and len(lines) == 1 and lines[0].startswith("FAILED "), (arguments, code, lines)
                assert "{}: no report".format(stem) in lines[0], (arguments, lines[0])
        assert not os.path.exists(os.path.join(root, "runs", "p27")), "a refused stage writes no selection"
