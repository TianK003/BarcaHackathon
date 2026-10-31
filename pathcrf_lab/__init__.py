"""P20 step 0 (R022): PathCRF's published checkpoint on its held-out DFL match, a measurement kept out of the pipeline.

Kim et al. (KDD 2026) publish a model that infers the ball's holder from player tracking alone. Before anyone spends
GPU time retraining it for us, one question: on J03WR9, the one DFL match it never trained or validated on, does its
team accuracy on the frames where our degraded ball is missing beat P07's held-out fold by the declared margin? The
answer is a measurement, so nothing here is imported by `pipeline/`, `evaluation/`, `starter/`, `games.py` or either
selftest, and a check enforces it: a measurement that fed the pipeline would stop being one, and torch must never
reach the pinned requirements of the submission.

Three processes exchange .npz files under data/cache/pathcrf/ (gitignored), so that torch lives in one venv only:

    adapter.py  (.venv)           serves the degraded match as the benchmark does, decodes main.toml, computes P07's
                                  fold, turns the 46 anonymous slots into complete nodes per 10 s window
    runner.py   (.venv-pathcrf)   PathCRF's own code, pinned by commit under data/external/pathcrf/repo/, on the
                                  clean match (the correctness check) and on each degraded window
    score.py    (.venv)           the only reader of the truth: accuracies, the reproduction gate, the go/no-go

Data rules, as everywhere (CLAUDE.md): the DFL data is public but still printed as aggregates only, the Calibration
and every degraded Atd carry BAR-ATH's masks and counts and are never printed, and nothing of BAR-ATH goes near
PathCRF. PathCRF's files are never committed. Every module imports torch inside functions only, so that the main
.venv can import them all.
"""
