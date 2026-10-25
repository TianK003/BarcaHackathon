# Stage 5 — output

**The point of this stage:** the deliverable is a file that our repository regenerates *byte for byte* from the data
(SUBMISSION.md, RULES.md). So this stage owns the bytes. It writes the CSV in exactly one way, checks it exactly the
way the organisers will, says what the labels claim, and records what produced them.

## In → out

| in | out |
|---|---|
| the labels of the whole match: `(n,)` int8 codes from stages 3–4, `X` outside the graded halves | the CSV, by default `evaluation/results/<config name>.csv` |
| the config, the inputs' source hashes, the git state | `data/runs/<name>.json` — the run manifest |
| optionally a ground truth (`--score`) | scores and a row ready to paste into `docs/results.md` |

## The CSV

```
frame,label
0,X
1,X
...
11185,H
...
199318,X
```

- Exactly 199,319 rows, `frame` 0…199,318 in order, labels `H`/`A`/`D`/`X`, no blank cells.
- **LF line endings, written in binary mode.** Text mode on Windows writes CRLF, which changes the bytes — and the
  sha256 — from one machine to the next ([T21](../../docs/traps.md)).
- Written to `<name>.csv.tmp` and renamed into place, so an interrupted run never leaves half a submission behind.
- Then read back and checked with `possession_metrics.read_label_csv` and `validate_submission` — the same calls as
  `evaluation/validate_submission.py` and the official scorer — and the labels read back must equal the labels
  written.

## Where a CSV may go

| path | allowed | why |
|---|---|---|
| `evaluation/results/<name>.csv` | yes (default) | the scorer looks there, and `.gitignore` covers `evaluation/results/*.csv` |
| anywhere under `data/` | yes | ignored as a whole |
| outside the repository | yes | e.g. `--out ../<team_name>.csv` for the upload |
| `baseline_*.csv` anywhere | **no** | those names are the organisers' committed files, and the `.gitignore` exception would commit ours too ([T24](../../docs/traps.md)) |
| anywhere else inside the repository, including sub-folders of `evaluation/results/` | **no** | not covered by `.gitignore`, so a file of match-derived labels could be committed ([T25](../../docs/traps.md)) |

## What gets printed

Aggregates only, never a frame's data. For `main.toml` as of R012 (the figures change whenever `main.toml` does, and
[docs/results.md](../../docs/results.md) has the current ones):

```
output    evaluation/results/main.csv   0.6 s
          valid: 199,319 rows   sha256 3fd06da0…
          graded frames: H 37.5%  A 26.6%  D 36.0%  X  0.0%
          439 possessions in the graded halves, mean 9.3 s
```

Possessions are counted the way the scorer counts them: inside the graded halves, with `X` bridged by the label
before it ([T06](../../docs/traps.md)). `validate_submission.py` counts differently (over the whole file, `X` as a gap),
so compare like with like.

With `--score` (the public window by default, or `--score path/to/labels.csv` for any other `frame,label` file of
this match, such as another run's output; there are no hand labels of this match, [T27](../../docs/traps.md)):

```
score     evaluation/gt/BAR-ATH_possession_gt_public.csv: macroF1 0.7159  spellF1 0.4000  ...  possessions 20 (truth 17)
          controlled_touch     home agrees on 87.2% of 5,489 frames   alive agrees on -- (no opinion ...)
          dead_ball_model      home agrees on -- (no opinion ...)   alive agrees on 86.6% of 7,500 frames
          shape_possession     home agrees on 90.9% of 6,031 frames   alive agrees on -- (no opinion ...)
results   | R0xx | 2026-10-25 | `main` @ <commit> | 0.7159 | 0.4000 | 20 / 14.1 s | | |
```

The public window checks correctness. It cannot rank versions: it prefers `smooth125` to `smooth50`, the reverse of
the full match ([T01](../../docs/traps.md)).

## The run manifest

`<data dir>/runs/<name>.json` — under `data/`, so never committed — named after the CSV it describes, so
`--out ../team_name.csv` writes `runs/team_name.json` and the last run to write a file is the one on record. It holds
the config file's text, the git commit
and whether the tree was dirty (untracked files count: a new component that is not committed makes a run
irreproducible from its commit), Python/numpy/pandas versions, the sha256 of both ATD source files, the output's
sha256, the summary, and the scores if `--score` was given. Timings are printed, not stored, so two manifests of the
same run compare equal.

## Before you upload

1. `python pipeline/run.py pipeline/configs/main.toml --out ../<team_name>.csv` on a clean checkout.
2. `python evaluation/validate_submission.py ../<team_name>.csv` — the organisers' own check, on the exact file.
3. Note the sha256 in the submission log in [docs/results.md](../../docs/results.md) and in the technical document.
