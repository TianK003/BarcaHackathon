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
    """Put their repository at the end of sys.path, so that `models`, `datatools` and `set_transformer` import."""
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
    """Their protocol on one match, as the tutorial runs it; returns the scores (team_acc is the correctness gate).

    The steps are the tutorial's cells: kloppy's sportec loader through their SportecData (only_alive=False), their
    Preprocessor and touches, the running features, written to parquet and read back as the tutorial does, then
    `inference` (CRF, decode="indep", 10 s windows, correct_episode_lasts), `detect_events`, and the metric of cells 21
    and 25: per-frame team accuracy of the event-split possession, truth events against predicted events.
    """
    import time
    import pandas as pd
    torch = prepare_torch()
    _repo(repo_dir)
    model = load_model(repo_dir)
    import datatools.sportec_data as sportec_data
    for name in ("META_DIR", "EVENT_DIR", "TRACKING_DIR"):
        setattr(sportec_data, name, dfl_dir)  # the XML is read: <TRACKING_DIR>_parquet does not exist
    from datatools.sportec_data import SportecData
    from datatools.preprocess import Preprocessor
    from datatools.utils import calculate_running_features
    from datatools.postprocess import detect_events
    from datatools.evaluate_events import align_true_and_pred, simplify_events
    from applications.team_poss import compute_true_pred_poss
    from inference import inference

    prints = 0
    events = pd.read_parquet(os.path.join(repo_dir, "data", "sportec_synced", match_id + ".parquet"))
    data, n = _quiet(SportecData, match_id, load_tracking=True)
    prints += n
    tracking = data.tracking.reset_index(drop=True)
    for column in ("timestamp", "ball_x", "ball_y"):
        tracking[column] = tracking[column].round(2)

    pre = Preprocessor(events, tracking)
    touches, n = _quiet(pre.detect_touches)
    prints += n
    merged, n = _quiet(pre.merge_tracking_poss, touches)
    prints += n
    # calculate_running_features: per player per period at 25 Hz, as the tutorial applies it to the merged tracking.
    processed, n = _quiet(calculate_running_features, merged)
    prints += n
    os.makedirs(work_dir, exist_ok=True)
    processed.to_parquet(os.path.join(work_dir, "tracking_processed.parquet"))
    events.to_parquet(os.path.join(work_dir, "event_processed.parquet"))
    tracking = pd.read_parquet(os.path.join(work_dir, "tracking_processed.parquet"))
    events = pd.read_parquet(os.path.join(work_dir, "event_processed.parquet"))

    started = time.perf_counter()
    (micro_pred, stats), n = _quiet(inference, model, tracking, use_crf=True, decode="indep",
                                    window_seconds=WINDOW_SECONDS, correct_episode_lasts=True)
    prints += n
    seconds = time.perf_counter() - started
    pred_events, n = _quiet(detect_events, micro_pred, tracking)
    prints += n

    true_events = simplify_events(events)
    frames = compute_true_pred_poss(true_events, pred_events, tracking)
    team_acc = float((frames.true_team == frames.pred_team).mean())
    scores, n = _quiet(align_true_and_pred, events, pred_events, threshold=1.0)
    prints += n
    total = scores.loc["total"]

    result = {"match": match_id, "team_acc": team_acc, "edge_acc": float(stats["edge_acc"]),
              "src_acc": float(stats["src_acc"]), "dst_acc": float(stats["dst_acc"]),
              "n_frames": int(stats["n_frames"]), "ev_precision": float(total["precision"]),
              "ev_recall": float(total["recall"]), "ev_f1": float(total["f1"]), "inference_s": round(seconds, 1),
              "their_prints": prints, "torch": torch.__version__, "threads": torch.get_num_threads()}
    np.savez_compressed(os.path.join(work_dir, "clean.npz"),
                        frame_number=micro_pred["frame_id"].to_numpy(),
                        team=codes_from_senders(micro_pred["edge_src"].to_numpy()),
                        alive=(micro_pred["ball_state"] == "alive").to_numpy(),
                        episode=micro_pred["episode_id"].to_numpy())
    return result


def smoke(repo_dir):
    """The real checkpoint on one seeded synthetic window of 22 random walkers; prints counts only. Returns 0."""
    torch = prepare_torch()
    model = load_model(repo_dir)
    print("torch {} threads {} keys matched (strict load, weights_only)".format(
        torch.__version__, torch.get_num_threads()))
    _repo(repo_dir)
    import pandas as pd
    from datatools.utils import calculate_running_features  # noqa: F401  (their kinematics on the walkers)
    from inference import inference_episode
    rng = np.random.default_rng([0, 22])
    steps = int(WINDOW_SECONDS * FPS)
    start = rng.uniform([10, 10], [95, 58], size=(22, 2))
    walk = start[:, None, :] + np.cumsum(rng.normal(0, 0.05, size=(22, steps, 2)), axis=1)
    print("synthetic window: {} nodes x {} frames, seeded; calling their inference_episode".format(*walk.shape[:2]))
    out = inference_episode(model, pd.DataFrame(walk.reshape(22, -1).T), use_crf=True, decode="indep",
                            window_seconds=WINDOW_SECONDS, last_node=None)
    senders = None if out is None else getattr(out, "edge_src", None)
    if senders is None:
        print("inference_episode returned {}: the smoke call is wrong for this API; read their inference.py".format(
            "None" if out is None else type(out).__name__))
        return 1
    codes = codes_from_senders(list(senders))
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
    result = clean(args.match, dfl_dir=os.path.join(data, "external", "dfl"), repo_dir=repo_dir, work_dir=work)
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
