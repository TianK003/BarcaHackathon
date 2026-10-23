# [Team name] — More than a Hack 2027, Phase 1: technical document

> **Draft.** This is deliverable 3 of SUBMISSION.md: a few pages, uploaded as `<team_name>.pdf` or `<team_name>.md`.
> Fill each section as decisions land, so that the last week is editing rather than writing. Delete these notes
> before submitting.

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `<team_name>.csv`, sha256 `<…>`
- **Scores we can state:** public window macroF1 … / spellF1 …; our validation …. The full-match score is the
  organisers'.

## 1. Approach

<!-- What we built, and why this rather than the alternatives. Link the proposals we tried (docs/proposals.md). -->

**Pipeline** (`pipeline/README.md`). Five stages: inputs → evidence → decode → refine → output. *Evidence* sources
look at the whole match and give, per frame, the probability that the ball is in play (`alive`) and that the home
team has it given play (`home`), or abstain. A *decoder* turns them into labels one graded half at a time, so nothing
is smoothed across halftime. *Refiners* apply metric-aware clean-ups, and the *output* stage writes the CSV with LF
line endings and validates it with the official checks. Splitting "what the data says" from "what the label sequence
should be" lets each source be measured on its own and lets one decoder weigh them all. The organisers' baseline
runs through the same pipeline and is rebuilt byte for byte as a regression check (R001).

## 2. Assets used and dropped

| asset | used? | how — or why it was dropped |
|---|---|---|
| Tactical ATD (players, ball, speeds) | | |
| Smart Tagging | | |
| Tactical homography | | |
| Tactical video (frames, audio) | | |
| Broadcast video, ATD, homography | | |
| Public 5-minute ground truth | | |
| Labelled tracking of other matches (DFL/IDSSE open data; P01, P13) | | |
| Other external datasets or pretrained models | | |

No frame of this match was hand-labelled, for training, validation or the submission (RULES.md, as clarified on
22 Oct 2026).

## 3. Missing and noisy tracking

<!-- Ball gaps (40.7% of graded frames, in long gaps — docs/results.md A01), fragmented tracklets, identity switches,
duplicates. What we did about each. -->

## 4. Possession rules implemented

- **Change of possession:** <!-- what counts as controlled possession -->
- **Loose balls and contested phases:**
- **Touches, deflections, aerial duels:**
- **Dead ball — when it starts and ends:**
- **Restarts — who gets the ball:**
- **Abstentions (`X`):**

## 5. Validation

<!-- How we chose between versions: the public window for correctness only; the P01 benchmark of other matches
(which matches, how they were degraded, how well the degraded copy reproduces our data's statistics, and whether it
ranks the four baselines in their published order); per-match t-intervals and an exact sign-flip test, never a
bootstrap (seven matches are not a sample to resample). What the benchmark could not measure (Smart Tagging, camera,
audio) and how we checked those instead. -->

## 6. Limitations

<!-- Where it fails, and what we would fix with more time. -->

## 7. Reproduction

```bash
python3.11 -m venv .venv                     # Windows: py -3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-pipeline.txt
# unpack the data bundle into data/ (see data/README.md)
.venv/bin/python pipeline/selftest.py --data                    # optional: rebuilds the organisers' 4 baselines byte for byte
.venv/bin/python pipeline/run.py pipeline/configs/main.toml     # writes evaluation/results/main.csv and validates it
.venv/bin/python evaluation/validate_submission.py evaluation/results/main.csv
```

Expected sha256 of `main.csv`: `<…>` (the run prints it). Runtime: about 20 s the first time, which includes parsing
the tracking into `data/cache/` once, then a few seconds per run, on [hardware].
