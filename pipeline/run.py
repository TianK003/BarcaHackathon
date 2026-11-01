"""One command from data/ to a validated submission CSV.

    python pipeline/run.py pipeline/configs/main.toml                  # -> evaluation/results/main.csv
    python pipeline/run.py pipeline/configs/main.toml --score          # + public-window scores, per-source agreement
    python pipeline/run.py CONFIG --score evaluation/results/other.csv  # ... against another run, to compare two
    python pipeline/run.py CONFIG --out ../team_name.csv                # the deliverable, outside the repository

What happens, in order -- each stage is a folder of pipeline/ with a README.md that explains it:

    1. config     the TOML file, checked before anything is loaded              pipeline/config.py
    2. inputs     the match, aligned to submission rows; ATD cached once         pipeline/inputs/
                  with [inputs] ball_fill = "broadcast", our ball's gaps filled from the broadcast's (R028)
    3. evidence   every [[evidence]] source over the whole match, checked        pipeline/evidence/
    4. decode     the [decoder], once per block -- each graded half              pipeline/decode/
    5. refine     every [[refine]] step in order, per block                      pipeline/refine/
    6. output     the CSV, validated with the official checks, summarised        pipeline/output/
    7. manifest   <data dir>/runs/<name>.json: config, commit, versions, hashes, scores

Frames outside every block are written as X: they are never graded, so an X there is free. The printout is aggregates
only -- shares, counts, hashes and scores -- never a frame's data (CLAUDE.md, hard rules).
"""

import argparse
import datetime
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if __name__ == "__main__":
    # Run by path, Python put pipeline/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no folder in here can shadow a top-level module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline import decode, evidence, output, refine  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import (X, Block, BlockContext, Evidence, RunContext, check_decoded,  # noqa: E402
                                check_evidence)
from pipeline.inputs import Inputs  # noqa: E402
from pipeline.inputs.broadcast import AlignmentError  # noqa: E402


@dataclass(frozen=True, eq=False)
class RunResult:
    codes: np.ndarray  # (n,) int8, the labels as written
    sha256: str        # of the CSV's bytes -- the number to quote in the submission log
    summary: dict      # output.summarise(): label counts, possessions
    evidence: dict     # name -> Evidence over the whole match, for --score and for notebooks
    timings: dict      # stage -> seconds (printed, not kept in the manifest: it would differ on every run)
    fill: dict = None  # the inputs' fill_summary when [inputs] ball_fill fills the ball, else None (aggregates only)


def prepare_inputs(config, inputs):
    """The inputs the config's components read: `inputs` itself, or a view of them with the ball filled as the config's
    [inputs] ball_fill says (Inputs.with_ball_fill). run() and the benchmark's predict() call it before any source runs,
    so the fill is applied in this one place; for ball_fill = "none" it is the very object, and nothing changes."""
    return inputs.with_ball_fill(config.ball_fill)


def make_blocks(scope, inputs):
    """The stretches decoded on their own: one per graded half, or the whole file for scope = "file"."""
    if scope == "file":
        return [Block(half=None, start=0, stop=inputs.n_frames)]
    return [Block(half=half, start=start, stop=stop) for half, start, stop in inputs.graded]


def compute_evidence(config, run_context):
    """name -> Evidence for every [[evidence]] entry, in config order, each checked against the contract."""
    sources = {}
    for component in config.evidence:
        source = evidence.REGISTRY[component.kind](run_context, **component.params)
        if not isinstance(source, Evidence):
            raise ValueError("evidence {!r} ({}) returned {}, not an Evidence".format(
                component.name, component.kind, type(source).__name__))
        source = replace(source, name=component.name, weight=component.weight)
        check_evidence(source, run_context.inputs.n_frames)
        # Read-only from here on: decoders and refiners get views of these arrays, so an in-place np.clip(...,
        # out=...) would otherwise rewrite the evidence every later block, refiner and --score reads.
        source.alive.flags.writeable = False
        source.home.flags.writeable = False
        sources[component.name] = source
    return sources


def decode_blocks(config, run_context, sources, blocks):
    """(n,) int8 labels: decoder then refiners, block by block; X on every frame outside the blocks.

    Each block's context holds every source cut down to that block -- the only evidence a decoder or refiner ever
    sees -- which is how the pipeline keeps the per-half rule (docs/traps.md T11) on everyone's behalf.
    """
    codes = np.full(run_context.inputs.n_frames, X, dtype=np.int8)
    for block in blocks:
        ctx = BlockContext(run=run_context, block=block,
                           evidence={name: source.window(block.start, block.stop) for name, source in sources.items()})
        decoded = decode.REGISTRY[config.decoder.kind](ctx, **config.decoder.params)
        check_decoded(decoded, block, "decoder {!r}".format(config.decoder.kind))
        for component in config.refine:
            decoded = refine.REGISTRY[component.kind](ctx, decoded, **component.params)
            check_decoded(decoded, block, "refiner {!r}".format(component.kind))
        codes[block.start:block.stop] = decoded.labels
    return codes


def run(config, inputs, out_path, log=print):
    """Run one config end to end and write the CSV to out_path. Returns a RunResult; raises on any broken contract."""
    timings = {}
    # Legacy global generators seeded too, for library code that reaches for them; components should derive their
    # own generator from run.seed (pipeline/contracts.py, RunContext).
    random.seed(config.seed)
    np.random.seed(config.seed)

    log("config    {}   scope {}, seed {}".format(_shown(config.path), config.scope, config.seed))
    if config.scope == "file":
        log("WARNING   scope = \"file\" decodes across halftime and over ungraded frames (docs/traps.md T11). "
            "Only the baseline reproduction should do that.")

    started = time.perf_counter()
    inputs.atd  # every real config needs the tracking; loading it here keeps the timings honest
    timings["inputs"] = time.perf_counter() - started
    log("inputs    ATD cache {:<8} {}   {:.1f} s".format(
        inputs.atd_status, _shown(os.path.join(inputs.cache_dir, "atd_tactical")), timings["inputs"]))

    inputs = prepare_inputs(config, inputs)
    run_context = RunContext(inputs=inputs, seed=config.seed)
    started = time.perf_counter()
    fill = inputs.fill_summary  # None without a fill; with one, the broadcast is loaded, aligned and filled in here
    if fill is not None:
        timings["fill"] = time.perf_counter() - started
        log("fill      {} ball: {}   {:.1f} s".format(config.ball_fill, _fill_line(fill), timings["fill"]))

    started = time.perf_counter()
    sources = compute_evidence(config, run_context)
    timings["evidence"] = time.perf_counter() - started
    for row in evidence.describe(sources, inputs.graded_mask):
        log("evidence  {:<20} opinion on graded frames: alive {:5.1%}  home {:5.1%}".format(
            row["name"], row["alive_coverage"], row["home_coverage"]))
    log("          {} source(s)   {:.1f} s".format(len(sources), timings["evidence"]))

    started = time.perf_counter()
    blocks = make_blocks(config.scope, inputs)
    codes = decode_blocks(config, run_context, sources, blocks)
    timings["decode"] = time.perf_counter() - started
    log("decode    {} over {} block(s){}   {:.1f} s".format(
        config.decoder.kind, len(blocks),
        ", then " + " -> ".join(component.kind for component in config.refine) if config.refine else "",
        timings["decode"]))

    started = time.perf_counter()
    sha256 = output.write_submission(out_path, codes)
    output.validate(out_path, inputs.n_frames, codes)
    summary = output.summarise(codes, inputs.graded, inputs.fps)
    timings["output"] = time.perf_counter() - started
    graded = summary["labels_graded"]
    log("output    {}   {:.1f} s".format(_shown(out_path), timings["output"]))
    log("          valid: {:,} rows   sha256 {}".format(summary["rows"], sha256))
    log("          graded frames: {}".format("  ".join("{} {:5.1%}".format(
        letter, graded[letter] / float(summary["graded_frames"] or 1)) for letter in ("H", "A", "D", "X"))))
    log("          {} possessions in the graded halves, mean {:.1f} s".format(
        summary["possessions"], summary["mean_possession_s"]))
    return RunResult(codes=codes, sha256=sha256, summary=summary, evidence=sources, timings=timings, fill=fill)


def inputs_record(inputs, fill):
    """The manifest's "inputs": the tactical ATD's sources and cache version, and with a fill a "broadcast" entry too.

    `fill` is the run's RunResult.fill. The entry holds its aggregates (the alignment, the live share, the frames
    filled), the broadcast ATD cache's sources and version, and the broadcast homography file's size and sha256: what
    the filled run was made from, beside what any run is made from. Without a fill the record is what it always was, so
    a config without [inputs] writes the manifest it did.
    """
    record = {"atd_sources": inputs.atd_manifest["sources"], "atd_cache_version": inputs.atd_manifest["cache_version"]}
    if fill is not None:
        record["broadcast"] = dict(fill, atd_sources=inputs.broadcast_atd_manifest["sources"],
                                   atd_cache_version=inputs.broadcast_atd_manifest["cache_version"],
                                   homography_source=inputs.broadcast_homography_source)
    return record


def report_scores(result, inputs, gt, log=print):
    """Score against a ground truth and print the scores, each source's agreement, and a results.md row."""
    scores, gt_codes = output.score(result.codes, gt)
    log("score     {}: macroF1 {:.4f}  spellF1 {:.4f}  alive/dead F1 {:.4f}  possessions {} (truth {})".format(
        scores["ground_truth"], scores["macro_f1"], scores["spell_f1_mean"], scores["alive_dead_f1"],
        scores["n_pred_spells"], scores["n_gt_spells"]))
    if gt == "public":
        log("          a correctness check only: the public window ranks versions wrongly (docs/traps.md T01)")
    for row in evidence.describe(result.evidence, inputs.graded_mask, gt_codes):
        log("          {:<20} home agrees on {}   alive agrees on {}".format(
            row["name"], _agreement(row, "home"), _agreement(row, "alive")))
    return scores


def git_state():
    """{"commit", "dirty"} of the repository, or Nones without git. Dirty counts untracked files too: a new
    component that is not committed yet makes a run irreproducible from its commit just as surely as an edit."""
    try:
        def git(*arguments):
            return subprocess.run(("git",) + arguments, cwd=ROOT, capture_output=True, text=True, timeout=30,
                                  check=True).stdout.strip()
        return {"commit": git("rev-parse", "--short=12", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def results_row(stem, git, scores, gt):
    """A row for the scoreboard in docs/results.md, ready to paste. The ID and the verdict are yours to fill in."""
    what = "`{}` @ {}{}".format(stem, git["commit"] or "?", "+dirty" if git["dirty"] else "")
    poss = "{} / {:.1f} s".format(scores["n_pred_spells"], scores["mean_spell_duration_s"])
    if gt == "public":
        return "| R0xx | {} | {} | {:.4f} | {:.4f} | {} | | |".format(
            datetime.date.today().isoformat(), what, scores["macro_f1"], scores["spell_f1_mean"], poss)
    return "| R0xx | {} | {} | | | | {:.4f} | |".format(datetime.date.today().isoformat(), what,
                                                         scores["spell_f1_mean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="the run's TOML file, e.g. pipeline/configs/main.toml")
    parser.add_argument("--out", help="where to write the CSV (default: evaluation/results/<config name>.csv)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--rebuild-cache", action="store_true", help="re-parse the ATD even if its cache is fresh")
    parser.add_argument("--score", nargs="?", const="public", metavar="GT",
                        help="score the result: 'public' (the default when the flag is given) for the 5 minutes "
                             "in this repository, or a path to a frame,label CSV of this match such as another "
                             "run's output")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        inputs = Inputs(data_dir=args.data_dir, rebuild=args.rebuild_cache)
        out_path = os.path.abspath(args.out or output.default_out_path(config.stem))
        output.check_out_path(out_path, inputs.data_dir)
        result = run(config, inputs, out_path)
        scores = report_scores(result, inputs, args.score) if args.score else None
    except (ConfigError, output.UnsafeOutPath, output.SubmissionInvalid, output.ScoreError, OSError,
            AlignmentError) as error:
        # The expected failures -- a config to fix, an output path that git could pick up, data not unpacked yet, a
        # file the scorer would reject, a broadcast that does not align to fill from -- get a sentence. Anything else
        # is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1

    git = git_state()
    manifest_path = os.path.join(inputs.data_dir, "runs", os.path.splitext(os.path.basename(out_path))[0] + ".json")
    output.write_manifest(manifest_path, {
        "config": {"path": _shown(config.path), "text": config.text},
        "git": git,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "platform": platform.platform()},
        "inputs": inputs_record(inputs, result.fill),
        "output": {"path": _shown(out_path), "sha256": result.sha256},
        "summary": result.summary,
        "scores": scores,
    })
    print("manifest  {}".format(_shown(manifest_path)))
    if scores:
        print("results   {}".format(results_row(config.stem, git, scores, args.score)))
    return 0


def _shown(path):
    """A path as people read it: relative to the repository when inside it, absolute otherwise."""
    path = os.path.abspath(path)
    relative = os.path.relpath(path, ROOT) if os.path.splitdrive(path)[0] == os.path.splitdrive(ROOT)[0] else path
    return path if relative.startswith("..") else relative.replace(os.sep, "/")


def _agreement(row, channel):
    if row[channel + "_agreement"] is None:
        return "-- (no opinion on judged frames)"
    return "{:.1%} of {:,} frames".format(row[channel + "_agreement"], row[channel + "_judged_frames"])


def _fill_line(fill):
    """The fill's aggregates as the log shows them: per half the offset and the fine score, the live share, the frames
    filled. A ball given rather than aligned (the benchmark's simulated one) has no offsets and no live share."""
    parts = []
    if "offsets" in fill:
        halves = sorted(fill["offsets"])
        parts.append("offsets {} frames".format(" / ".join("{:+,}".format(fill["offsets"][half]) for half in halves)))
        parts.append("fine {}".format(" / ".join("{:.3f}".format(fill["fine_scores"][half]) for half in halves)))
    if "live_share" in fill:
        parts.append("live {:.1%} of the valid graded frames".format(fill["live_share"]))
    parts.append("{:,} frames filled ({:.1%} of the graded)".format(
        fill["filled_frames"], fill["filled_frames"] / float(fill["graded_frames"] or 1)))
    return ", ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
