"""The possession pipeline: one command from data/ to a validated submission CSV.

    python pipeline/run.py pipeline/configs/main.toml

Five stages, one folder each, every folder with a README.md that explains the stage:

    inputs  ->  evidence[]  ->  decode  ->  refine[]  ->  output

pipeline/README.md is the overview: the contracts between stages, the configs, and how to add a component.

Importing this package puts evaluation/ on sys.path, so `import possession_metrics` resolves exactly as it does
for the organisers' scripts -- the pipeline scores and validates with the official code, never a copy of it. The
repository root needs no such help: it is already on sys.path, because that is how `pipeline` itself was found.
"""

import os
import sys

# tomllib (the config reader) arrived in Python 3.11, and 3.11.9 is the team standard (docs/traps.md T20). Failing
# here with a sentence beats a ModuleNotFoundError three imports deep.
if sys.version_info < (3, 11):
    raise ImportError("the pipeline needs Python 3.11 or newer (found {}.{}); use the team venv, .venv".format(
        *sys.version_info[:2]))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION_DIR = os.path.join(ROOT, "evaluation")
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)
