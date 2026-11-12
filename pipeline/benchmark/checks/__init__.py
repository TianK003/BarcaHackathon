"""The benchmark's self-checks, one module per topic. pipeline/benchmark/selftest.py is the harness that runs them.

A check is a function named test_* that takes no arguments and raises if what it checks is wrong. It belongs in the
module of its topic:

    sources.py   the DFL registry and the download
    dfl.py       the DFL parsers and the cache of parsed matches (synthetic, and the downloaded ones with --dfl)
    hazard.py    the gap-onset hazard: features, logistic fit, AUC and the mask sampler (synthetic arrays only)
    calibration.py
                 the calibration, our own tracking measured for the benchmark: every statistic exact on hand-built
                 objects, a synthetic bundle, the cache, the hazard fit of our ball gaps (rows, AUCs, the go rule, the
                 duration cells), and with --data the figures of A01, A02 and R001 and A08's fit
    degrade.py   the degrader's shared parts: the transplant worked out exactly, the draws, what the gaps profile
                 touches and does not, that no truth reaches it, and that the order of the matches changes nothing
                 (synthetic; with --data our real masks on a downloaded DFL match)
    cap.py       the tracking profile's cap of the low-count tail (the rule worked out one row at a time, sticky
                 membership, the counts of a window of ours), its shortest-piece rule, the order of the cap, the
                 episodes and the smoothing, and the scale of the tracked spans (synthetic only)
    context.py   the context profile of the degrader: the tracking profile's players unchanged (no truth, the same
                 draws), and a ball whose gaps come from the hazard fitted on our match: what the hazard is given, the
                 chance (c, the scale, the cap, 0 where a feature is undefined), the class, third and durations read at
                 the at-risk frame, the counts of the record on hand-built halves, and the odds rescale of the scale
                 (synthetic; with --data our real Calibration on a downloaded DFL match)
    tracking.py  the tracking profile of the degrader: tracklets and slots (walks, lowest free slot, no overlap, the
                 degenerate case equal to gaps, the lost share on long runs), the wrong-team episodes, the NaN-speed
                 runs, the smoothing, and that the profiles nest (synthetic; with --dfl that no pool of slots runs out
                 on the 7 matches; with --data our real Calibration on a downloaded DFL match)
    generic.py   the generic profile of the degrader (P27), the benchmark's default, which reads nothing of our match:
                 its constants are the declared ones, no Calibration is taken or loaded (by the degrader, the runner,
                 the trainers or the instruments), no truth reaches it, the ball's mask is the documented draws (41%
                 missing in log-uniform gaps of 1 s to 60 s) laid from each kick-off, the players are lost at the
                 declared rate in 46 slots, and the runner reports it with no Calibration (synthetic only)
    fidelity.py  the fidelity report: every gate and check worked out by other code on a synthetic Calibration, the
                 pooling by sums, the file and the exit code (synthetic only)
    dfl_events.py
                 the DFL restart events on the timeline: the frame-clock mapping, the kick-off's team, what falls
                 outside the halves, what is refused (synthetic, and the seven matches with --dfl)
    truth.py     the benchmark's truth from the DFL flags: codes, summary, line timing, nearest team, the coarse
                 truth and the cost of a lag (synthetic, and the downloaded matches with --dfl)
    inputs.py    a DFL match served to the pipeline as MatchInputs and clean_atd: contract, no truth (synthetic, and
                 a downloaded match with --dfl)
    score.py     HalfScore, pooled and per_match against the official pm.evaluate, predict against pipeline.run.run,
                 the yardstick configs, and the match-level statistics (t_interval, sign_flip_p, compare)
                 (synthetic, and a downloaded match with --dfl)
    run.py       the benchmark runner: its reports, the reference and the direction of a comparison, the same bytes in
                 another process, what it refuses, both truths from one decode, and the oracle, which must score 1.0
                 on every real match (synthetic, and the downloaded matches with --dfl)
    choose.py    choose: a grid's rule worked out by hand from the runner's results, its deltas the runner's, the
                 leave-one-match-out picks in a plain loop, and the reports it refuses (synthetic only)
    tag_stats.py tag_stats: the agreement of Smart Tagging's votes with our own pseudo-labels, from synthetic votes
                 and episodes (synthetic only)
    headroom.py  the headroom runner: its six configs in order, the oracles that carry the truth and nothing else, that
                 they are in no production registry and the registry is as it was after a run (raised or not), the
                 variants as files that read back as the config with the oracles in, the coarse truth reaching them,
                 and that the full oracle scores exactly 1.0 (synthetic only)
    touch.py     touch_stats: its figures against a plain loop over the frames, the confidence rule, the grid and its
                 labels, and that the source is never handed the truth (synthetic only)
    dead_ball.py P05's dead-ball trainer: the exported trees against scikit-learn, the same bytes on 1 and 4
                 threads, the folds that exclude the held-out match, no truth in the features, the JSON artifact
                 (exact, LF, bounded), the logistic reference's bookkeeping, the rule values' rounding, and --verify
                 (synthetic only)
    conventions.py
                 the conventions instrument: the window rule against a plain loop, the detectors on hand-built
                 tracking, that the DFL mode reads the truth beside its inputs, that the public mode prints
                 aggregates and no transition frame, the bootstrap of the gap; and P02's pipeline latency and
                 declared shift rule: the typed matching at the tolerance, each condition of the rule on a list that
                 fails it, the out-of-fold prediction against shift_edges.apply, that a latency file of another
                 content of the config (or of none) is refused, and that nothing of BAR-ATH's is printed beyond
                 `limited` / `shown` / `interval_text` (synthetic only)
    restarts.py  restarts: the restarts found on a hand-built truth, the three outcomes, the ball shares of the predicted
                 stoppages, the pooling over matches and seeds, and that the truth never reaches the pipeline;
                 also `stats` (the confidences), `events` (the calls against DFL's restart events) and `configs`
                 (R014's and R014b's generated configs) (synthetic only)
    shape.py     P07's shape-possession trainer: the rows (every stride-th alive frame, 1 for H), the augmentation that
                 balances the classes exactly and mirrors every row, P05's settings with no monotone constraint, the
                 exported trees against scikit-learn's raw score, the same bytes on 1 and 4 threads, the folds that
                 exclude the held-out match, no truth in the features, the artifact (the source reads back what
                 scikit-learn says of the held-out match; exact, LF, bounded; retraining gives identical bytes), the
                 report (aggregates, worked out frame by frame, the stop rule, no match scored in sample), and
                 touch_agreement on a hand-built scene (synthetic only)
    label_diff.py
                 label_diff: the shares, possessions per half, changes in ball gaps and moved D onsets worked out on
                 hand-built codes, and that the printout holds figures and no frame (synthetic only)
    wrong_team.py
                 wrong_team: the over-11 and surplus rules, the candidate frames as controlled_touch's without its duel
                 test, the ratio at the rule's edge, the constants held equal to main.toml's, D3's onset ratio and the
                 printout (synthetic only)
    metrica.py   R023 (P16): Metrica's sample games, the pins, the fetch and the licence stop against a fake opener,
                 both readers, each integrity failure, the derivation worked out by hand, the decision at its
                 edges, and the drop and invalid rules (synthetic only)
    diagnose.py  A19's diagnose: the IoU ladder of a hand-built half worked out by hand (counts, classes, losses, edge
                 errors), its matched pairs the scorer's own at every IoU and its spellF1 score.pooled's, the
                 calibration's figures and a known miscalibration recovered, and diagnose() end to end: the declared
                 layout, the runner's spellF1, every measure's frames tallied, no long list, the file, what it refuses
                 (synthetic only)
    p27.py       P27's two tools (R030), on made-up reports and configs written to a temporary folder: p27_choose's
                 rules (the tie band's edge on the unrounded scores with R030's own margins, and what a mean exactly
                 0.005 below does; the stages' tie orders; Stage D's groups with the incumbent a candidate, the shape
                 source, the changes handed on; the shift's guard; the final bar; the exit codes and the NEXT line)
                 and p27_grid's configs (the declared grids under the stems that p27_choose reads, each loaded as the
                 runner loads it, the shift, the confidence read from the touch file, the prefix) (synthetic only)
    p29.py       p29_bar (the bar of P29 to P31 as a script) (synthetic only)
    p30.py       P30's three configs: p27_final but for the shape model and its weight (config files)
    p31.py       P31's config: p27_final but for the decoder's team_neutral_dead flag (config files)
    team.py      P32's team model where the ball is seen (train_team) and its four configs (synthetic and config files)
    p34.py       train_shape's report: the coarse-truth cells kept from P34 (dropped, R036) (synthetic only)
    harness.py   the harness itself: what it collects, how it reports NOT RUN, the console set-up, and that no check
                 module imports another one
    common.py    what several modules share, and no checks of its own: the synthetic DFL match, the random one
                 (random_match), the synthetic Calibration and the other fixtures that the modules above build on, the
                 leak detector (leaks) that finds the match or its truth in what a component's inputs reach,
                 real_match, damaged, NotRun

A new topic gets a new module here, listed in MODULES. The harness refuses a module of this folder that MODULES does
not list (it would run nothing) and an entry of MODULES that is not one of them (a production module of the same name,
say, which would hide that), comparing whole names. It collects the test_* functions that each module in MODULES
defines itself (not those it imports), and runs them sorted by name across all modules, which is the order the single
file always ran in; the name decides the rest: test_dfl_* needs the downloaded DFL files and runs only with --dfl,
test_data_* needs the real match and runs only with --data. A name used twice is refused, since the harness could not
tell the two apart.

A check that cannot run on this machine (an optional dependency is missing) raises common.NotRun(reason): the
harness prints NOT RUN with the reason and counts it apart from the passes. It never returns quietly, which would read
as a pass of something that was not compared.

Import with absolute imports (from pipeline.benchmark import dfl): a check module named like a production module
cannot then shadow it. A check module imports from common.py and from production code, and from no other check module;
what two of them share goes to common.py (test_a_check_module_imports_no_other_check_module).
"""

from pipeline.benchmark.checks import (
    calibration, cap, choose, context, conventions, dead_ball, degrade, dfl, dfl_events, diagnose, fidelity, generic,
    harness, hazard, headroom, inputs, label_diff, metrica, p27, p29, p30, p31, p34, restarts, run, score,
    shape, sources, tag_stats, team, tracking, touch, truth, wrong_team)

MODULES = (sources, dfl, dfl_events, truth, inputs, score, run, choose, hazard, calibration, degrade, cap, tracking, context,
           generic, fidelity, headroom, touch, dead_ball, conventions, restarts, shape, label_diff, wrong_team,
           tag_stats, metrica, diagnose, p27, p29, p30, p31, team, p34, harness)
