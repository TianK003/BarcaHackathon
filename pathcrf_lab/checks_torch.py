"""Checks that need torch and the fetched PathCRF repo; run with `.venv-pathcrf` and `selftest.py --torch`.

A stub until the runner exists. Every check raises NotRun when torch does not import or sources.verify() is not all
ok, so that a machine without them reports NOT RUN and never a pass.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathcrf_lab.selftest import NotRun  # noqa: E402


def _need_torch_and_repo():
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise NotRun("torch does not import here ({})".format(error))
    from pathcrf_lab import sources
    bad = {path: status for path, status in sources.verify().items() if status != "ok"}
    if bad:
        raise NotRun("{} of the pinned PathCRF files are not ok on disk; fetch them first".format(len(bad)))


def test_torch_and_repo_present():
    _need_torch_and_repo()


def test_one_thread_after_prepare():
    _need_torch_and_repo()
    from pathcrf_lab import runner
    torch = runner.prepare_torch()
    assert torch.get_num_threads() == 1, "{} torch threads after prepare_torch".format(torch.get_num_threads())


def test_load_model_weights_only_on_a_synthetic_checkpoint():
    _need_torch_and_repo()
    import tempfile
    from pathcrf_lab import runner, sources
    torch = runner.prepare_torch()
    repo = sources.repo_dir()
    runner._repo(repo)
    from models.utils import build_model, load_trial_args
    seeded = build_model(load_trial_args(os.path.join(repo, "saved", str(runner.TRIAL))), "cpu")
    with tempfile.TemporaryDirectory() as folder:
        good = os.path.join(folder, "good.pt")
        torch.save(seeded.state_dict(), good)
        model = runner.load_model(repo, checkpoint=good)
        mine, theirs = seeded.state_dict(), model.state_dict()
        assert mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine), \
            "the loaded weights are not the saved ones"
        bad = os.path.join(folder, "bad.pt")
        torch.save({"weights": object()}, bad)
        try:
            runner.load_model(repo, checkpoint=bad)
        except Exception:
            pass
        else:
            raise AssertionError("a checkpoint holding a pickled object was loaded under weights_only=True")


def _seeded_model():
    from pathcrf_lab import runner, sources
    torch = runner.prepare_torch()
    repo = sources.repo_dir()
    runner._repo(repo)
    from models.utils import build_model, load_trial_args
    torch.manual_seed(0)
    model = build_model(load_trial_args(os.path.join(repo, "saved", str(runner.TRIAL))), "cpu")
    model.eval()
    return model


def _right_home_frame():
    from pathcrf_lab import checks_main, runner
    return runner.window_frame(*checks_main._window_arrays(100))


def test_decode_window_one_code_per_row():
    _need_torch_and_repo()
    import numpy as np
    from pathcrf_lab import runner
    model = _seeded_model()
    frame = _right_home_frame()   # home is on the right: the model's left team is away
    senders = runner.decode_window_senders(model, frame)
    codes = runner.decode_window(model, frame)
    assert len(codes) == len(frame) and set(codes.tolist()) <= {0, 1, 2}, "one code in {0, 1, 2} per row"
    assert codes.tolist() == [runner.team_code(s) for s in senders], "a code must be team_code of the sender's name"
    assert np.array_equal(codes, runner.decode_window(model, frame)), "the same codes on a second call"


def test_decode_window_holds_each_step_five_frames():
    _need_torch_and_repo()
    from pathcrf_lab import runner
    codes = runner.decode_window(_seeded_model(), _right_home_frame()).tolist()
    changes = [i for i in range(1, len(codes)) if codes[i] != codes[i - 1]]
    assert all(i % 5 == 0 or i == len(codes) - 1 for i in changes), "codes changed off a 5-frame step: {}".format(changes)
