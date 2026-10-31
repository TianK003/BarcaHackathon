"""The runner, in .venv-pathcrf: PathCRF's own code, loaded from its pinned copy, on the clean match and on windows.

Why a runner of our own at all: the published model is only worth measuring if it is run the way its authors run it,
so the first thing this module does is their protocol on the clean J03WR9 (their preprocessing, their inference, their
metric) and checks the result against their number before any degraded window is touched. Their code is imported from
data/external/pathcrf/repo/ (pinned by commit, sources.py) and never copied into ours; only the glue is ours.

Torch and their modules are imported inside functions only, so that the torch-free helpers (the team of a sender's
name) import in the main .venv as well, where score.py uses them.

    python pathcrf_lab/runner.py smoke                    # the real checkpoint on one synthetic window; prints counts
    python pathcrf_lab/runner.py clean --match J03WR9     # their protocol on the clean match -> runs/pathcrf/clean.json

Nothing is printed from the match but aggregates: scores and counts. Their own prints are captured and counted.
"""

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    # Run by path, Python put pathcrf_lab/ itself first on sys.path: put the repository root there instead.
    sys.path[0] = ROOT

import games  # noqa: E402

# Params
TRIAL = 140
CHECKPOINT = "state_dict_best_acc.pt"
FPS = 25.0
SAMPLE_FREQ = 5
WINDOW_SECONDS = 10.0
PITCH = (105.0, 68.0)
CODES = {"home": 1, "away": 0}
OUT = 2
SILENT = -1


def team_code(sender):
    """1 for a home node, 0 for an away node, OUT for one of the four outside nodes: by the node's name, never by block.

    Their blocks are the left and the right team, not home and away, so only the name says whose a node is. The padding
    nodes (`home_pad_k`) are named for their team too. Anything else is a ValueError that names the value.
    """
    name = str(sender)
    if name.startswith("home_"):
        return CODES["home"]
    if name.startswith("away_"):
        return CODES["away"]
    if name.startswith("out_"):
        return OUT
    raise ValueError("a node name must start with home_, away_ or out_, got {!r}".format(sender))


def codes_from_senders(senders):
    """int8 team codes of a sequence of sender names; NaN or None (no output for that frame) is SILENT."""
    codes = np.full(len(senders), SILENT, dtype=np.int8)
    for i, sender in enumerate(senders):
        if sender is None or (isinstance(sender, float) and np.isnan(sender)):
            continue
        codes[i] = team_code(sender)
    return codes


def prepare_torch():
    """One thread for torch, intra and inter-op, and a seed; asserts that the setting took. Returns the torch module."""
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set once in this process, which is the one thing we ask for
    torch.manual_seed(0)
    assert torch.get_num_threads() == 1, "torch uses {} threads, expected 1".format(torch.get_num_threads())
    return torch


def _repo(repo_dir):
    """Put their repository at the end of sys.path, so that `models`, `datatools` and `set_transformer` import.

    No bytecode is written, so that importing their modules leaves the pinned tree as fetched.
    """
    sys.dont_write_bytecode = True
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)


def load_model(repo_dir, checkpoint=None):
    """Trial 140's model in eval mode on the CPU, with the checkpoint's weights loaded strictly and weights_only.

    `checkpoint` overrides the path of the state dict (the checks save a seeded one of their own). weights_only=True
    refuses anything but tensors, so a file that is not a plain state dict never runs its code.
    """
    torch = prepare_torch()
    _repo(repo_dir)
    from models.utils import build_model, load_trial_args
    saved = os.path.join(repo_dir, "saved", str(TRIAL))
    model = build_model(load_trial_args(saved), "cpu")
    path = checkpoint or os.path.join(saved, "model", CHECKPOINT)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _quiet(call, *args, **kwargs):
    """call(*args, **kwargs) with their stdout captured; returns (result, number of lines they printed)."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        result = call(*args, **kwargs)
    return result, len(sink.getvalue().splitlines())


def clean(match_id, *, dfl_dir, repo_dir, work_dir):
    """Their protocol on one match, as datatools/preprocess.py's main block and the tutorial run it.

    `dfl_dir` holds the match's three DFL files (matchinformation, events_raw, positions_raw_observed). The steps:
    SportecData (kloppy, only_alive=False), Preprocessor on the synchronised events in the repository, touches, the
    possession merged into the tracking, the running features, both written to parquet and read back; then
    `inference` (CRF, decode="indep", 10 s windows, correct_episode_lasts), `detect_events`, and the metric: per-frame
    team accuracy of the event-split possession, truth touches against the events decoded from the path. Returns the
    scores; team_acc is the correctness gate.
    """
    import time
    import pandas as pd
    torch = prepare_torch()
    _repo(repo_dir)
    model = load_model(repo_dir)
    import datatools.sportec_data as sportec_data
    for name in ("META_DIR", "EVENT_DIR", "TRACKING_DIR"):
        setattr(sportec_data, name, dfl_dir)  # read at call time; <TRACKING_DIR>_parquet does not exist: the XML is read
    from applications.team_poss import compute_true_pred_poss
    from datatools.evaluate_events import align_true_and_pred
    from datatools.postprocess import detect_events, simplify_events
    from datatools.preprocess import Preprocessor
    from datatools.sportec_data import SportecData
    from datatools.utils import calculate_running_features
    from inference import inference

    prints = 0
    synced = pd.read_parquet(os.path.join(repo_dir, "data", "sportec_synced", match_id + ".parquet"))
    data, n = _quiet(SportecData, match_id, load_tracking=True)
    prints += n
    tracking = data.tracking.reset_index(drop=True)
    tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)

    pre = Preprocessor(synced, tracking)
    touches, n = _quiet(pre.detect_touches)
    prints += n
    merged = pre.merge_tracking_poss(touches)
    processed, n = _quiet(calculate_running_features, merged)
    prints += n
    os.makedirs(work_dir, exist_ok=True)
    processed.to_parquet(os.path.join(work_dir, "tracking_processed.parquet"))
    touches.to_parquet(os.path.join(work_dir, "event_processed.parquet"))
    tracking = pd.read_parquet(os.path.join(work_dir, "tracking_processed.parquet"))
    events = pd.read_parquet(os.path.join(work_dir, "event_processed.parquet"))

    started = time.perf_counter()
    (_, _, micro_pred, stats), n = _quiet(inference, model, tracking, use_crf=True, decode="indep",
                                          correct_episode_lasts=True, window_seconds=WINDOW_SECONDS, fps=FPS,
                                          sample_freq=SAMPLE_FREQ)
    prints += n
    seconds = time.perf_counter() - started
    pred_events, n = _quiet(detect_events, tracking, micro_pred)
    prints += n

    frames = compute_true_pred_poss(simplify_events(events), pred_events, tracking)
    team_acc = float((frames["true_team"] == frames["pred_team"]).mean())
    (scores, _), n = _quiet(align_true_and_pred, events, pred_events, threshold=1.0)
    prints += n
    total = scores.loc["total"]

    result = {"match": match_id, "team_acc": team_acc, "edge_acc": float(stats.get("edge_acc", float("nan"))),
              "src_acc": float(stats.get("src_acc", float("nan"))),
              "dst_acc": float(stats.get("dst_acc", float("nan"))), "n_frames": int(stats["n_frames"]),
              "ev_precision": float(total["precision"]), "ev_recall": float(total["recall"]),
              "ev_f1": float(total["f1"]), "inference_s": round(seconds, 1), "their_prints": prints,
              "torch": torch.__version__, "threads": torch.get_num_threads()}
    ordered = tracking.set_index("frame_id").sort_index()
    np.savez_compressed(os.path.join(work_dir, "clean.npz"),
                        frame_number=ordered.index.to_numpy(),
                        team=codes_from_senders(micro_pred["edge_src"].reindex(ordered.index).to_numpy()),
                        alive=(ordered["ball_state"] == "alive").to_numpy(),
                        episode=ordered["episode_id"].to_numpy())
    return result


def synthetic_window(seed=0, frames=250):
    """A seeded window of 22 random walkers in the preprocessed layout their inference_episode reads.

    Columns `<team>_<k>_{x,y,vx,vy,speed,accel}` for 11 home and 11 away players (home_0 and away_0 are the keepers,
    kept at the two ends), no NaN, no outside nodes (their code adds them). Velocities are finite differences at 25 Hz:
    the layout is what is under test, not the smoothing. Imports nothing of theirs.
    """
    import pandas as pd
    rng = np.random.default_rng([seed, 22])
    steps = np.cumsum(rng.normal(0.0, 0.05, size=(frames, 22, 2)), axis=0)
    start = rng.uniform([10.0, 10.0], [95.0, 58.0], size=(22, 2))
    start[0] = (5.0, 34.0)
    start[11] = (100.0, 34.0)
    xy = np.clip(start[None] + steps, 0.0, [PITCH[0], PITCH[1]])
    vel = np.gradient(xy, 1.0 / FPS, axis=0)
    speed = np.linalg.norm(vel, axis=2)
    accel = np.abs(np.gradient(speed, 1.0 / FPS, axis=0))
    columns = {}
    for i in range(22):
        name = "{}_{}".format("home" if i < 11 else "away", i % 11)
        for suffix, values in (("x", xy[:, i, 0]), ("y", xy[:, i, 1]), ("vx", vel[:, i, 0]), ("vy", vel[:, i, 1]),
                               ("speed", speed[:, i]), ("accel", accel[:, i])):
            columns["{}_{}".format(name, suffix)] = values
    return pd.DataFrame(columns)


def smoke(repo_dir):
    """The real checkpoint on one seeded synthetic window, through their inference_episode; counts only. Returns 0."""
    torch = prepare_torch()
    model = load_model(repo_dir)
    print("torch {} threads {} keys matched (strict load, weights_only)".format(
        torch.__version__, torch.get_num_threads()))
    from inference import inference_episode
    window = synthetic_window()
    _, _, micro, order, _ = inference_episode(
        model, window, left_gk="home_0", right_gk="away_0", use_crf=True, decode="indep", last_node=None,
        evaluate=False, window_seconds=WINDOW_SECONDS, fps=FPS, sample_freq=SAMPLE_FREQ, device="cpu")
    codes = codes_from_senders(micro["edge_src"].to_numpy())
    print("{} nodes in order, {} frames decoded".format(len(order), len(micro)))
    print("decoded team codes: " + ", ".join("{}={}".format(name, int((codes == value).sum()))
                                             for name, value in (("home", 1), ("away", 0), ("out", OUT),
                                                                 ("silent", SILENT))))
    return 0


def main(argv=None, data_dir=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "clean"):
        one = sub.add_parser(name)
        one.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
        if name == "clean":
            one.add_argument("--match", default="J03WR9", help="the DFL match to run clean (default J03WR9)")
    args = parser.parse_args(argv)
    data = args.data_dir or data_dir or games.DATA_DIR
    repo_dir = os.path.join(data, "external", "pathcrf", "repo")
    if args.command == "smoke":
        return smoke(repo_dir)
    work = os.path.join(data, "cache", "pathcrf", args.match, "clean")
    dfl_dir = os.path.join(data, "external", "dfl", args.match)
    result = clean(args.match, dfl_dir=dfl_dir, repo_dir=repo_dir, work_dir=work)
    for key, value in result.items():
        print("{:<13} {}".format(key, value))
    out = os.path.join(data, "runs", "pathcrf")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "clean.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
