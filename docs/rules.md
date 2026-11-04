# Rules: the organisers' clarification of November 2026

**This file is the team's statement of the rules, since 3 Nov 2026.** The organisers emailed every participant a
"Clarification of the rules" (verbatim below). They say it "take[s] precedence over any other information you may have
received previously, whether in the documentation, in earlier emails or in answers to individual questions". So where it
differs from RULES.md, FAQ.md, EVALUATION.md, SUBMISSION.md, OVERVIEW.md or the clarifications of 22 and 23 Oct
(upstream `c073f29`, `f8f2629`), the email wins, and it also wins over anything our own docs said before 3 Nov. Where
the email says nothing, those documents still hold. That covers rule 3 (the repository regenerates the same file), the
data licence accepted at enrolment, the deliverables and the conditions on what we may use (the organisers can access
it, the technical document explains it).

Upstream has no commit carrying the email yet: on 3 Nov `git fetch upstream` still ended at `f8f2629`, and on 4 Nov it
still does. When one lands, compare it with this file and update both.

## The clarification, verbatim

> **Clarification of the rules**
>
> Dear participants,
>
> There has been some confusion about what is and what is not permitted in the Initial Challenge, so we are sending
> this clarification to all of you.
>
> Please note that the rules below take precedence over any other information you may have received previously,
> whether in the documentation, in earlier emails or in answers to individual questions.
>
> **What is not allowed**
>
> - Checking your output against the video.
> - Hand labels, human-produced data or any other data created manually on the provided game.
> - Uploading the provided data to AI services.
> - Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video.
>
> **What is allowed**
>
> - Running automatic checks using only the provided data.
> - Inspecting the provided data and the video to better understand what counts as possession.
> - Using as input the video, the automatic tracking data, the smart tagging or any other generated data, as long as
>   your algorithm was not trained on the ground truth or on the provided video.
> - Providing files of intermediate steps.
> - Using any dataset you have the rights to use. Please check the license of any dataset before using it.
>
> **Training and tuning**
>
> If you need to train (or retrain), you MUST use only external data. If your algorithm has parameters that need to be
> tuned, you MUST tune them on external data, never on the ground-truth possession data or on the provided video.

## What it changes

"Before" is RULES.md as clarified on 22 and 23 Oct, which our docs followed until 3 Nov.

| topic | before (RULES.md, `c073f29`, `f8f2629`) | since 3 Nov |
|---|---|---|
| Learning on this match without human labels (unsupervised, self-supervised, pseudo-labels made by an algorithm) | allowed ("What remains fine") | **not allowed**: "If you need to train (or retrain), you MUST use only external data" |
| A constant measured on this match without labels (say, agreement with our own pseudo-labels) | allowed, unless it came from the public window | **not allowed**: parameters that need tuning are tuned "on external data" |
| Training, fitting or calibrating on the video | not named | **not allowed**, by name |
| Watching the video | "to understand them, debug your pipeline and design features" | **only** "to better understand what counts as possession"; "checking your output against the video" is not allowed |
| Automatic checks on the provided data | implied | **allowed**, by name |
| The public 5 minutes | a correctness check, "not training data and not a tuning set" | the same: no training, fitting or calibrating on "the ground-truth possession data"; scoring a file against it is an automatic check |
| Hand labels, human-produced data about this match | not allowed | not allowed, and nor is "any other data created manually on the provided game" |
| The data and AI | the T&C: never to a third-party or cloud LLM | "Uploading the provided data to AI services" is not allowed |
| Files of intermediate steps | not addressed | **allowed** to provide |
| Other datasets | other games' data, hand-labelled included, if the organisers can access it and the technical document explains it | "any dataset you have the rights to use", its licence checked |

## How we read it

Where the text leaves room we take the stricter reading. We ask the organisers (below) before relying on a looser one.

1. **The provided game is test data and nothing else.** Our algorithm reads its tracking, tagging, homography, video,
   audio and ground truth at run time, and our checks read them too. Nothing is trained, retrained, fitted, calibrated,
   adapted or tuned on any of them, with labels or without: no pseudo-labels, no self-training, no priors re-estimated
   on this match, no clusters fitted to its frames.
2. **Running is not fitting.** A fixed rule that reads this match and computes from it is the algorithm running. That
   covers a control episode, a team's mean position, which way a team attacks, or the Viterbi path of a decoder whose
   parameters were set elsewhere. Fitting is a number measured on this match and then written into a config, a model
   or a prior: a confidence, a weight, a threshold, a class prior, a scaler, a cluster centre. Such a number comes only
   from external data, from a fixed rule (the laws of the game, the literature), or from a fact the organisers
   document (the kick-off frames).
3. **A check passes or fails; it never chooses.** "Running automatic checks using only the provided data" is allowed.
   That covers the validator, determinism and the sha256, alignment, the `D` share and possession count of our own
   output, a transfer gate that can stop a model trained elsewhere (R011), and a score against the public window. A
   check may stop a candidate. It never ranks candidates or sets a value ([T01](traps.md)).
4. **Nobody checks our output against the video.** A person may watch the video and inspect the data to understand
   what counts as possession: where our ground truth starts and ends a possession or a stoppage. They may not hold our
   labels or transitions up against the video. That includes the scorer's `_transitions.csv`, a run's output, and the
   events our components detect (a controlled touch, `alive` crossing 0.5). Nothing they see becomes a label, a
   correction, or a rule or constant tied to frames ([T27](traps.md)). In P02's `public_transitions.csv`, watch the
   truth's changes and ignore the detectors' lag columns.
5. **No provided data goes to any AI service**, and our practice extends that to every LLM, Claude included, hosted or
   local. The public 5-minute ground truth (`evaluation/gt/`) and the organisers' baseline CSVs are labels of this
   match's frames, so we treat them as provided data too: an LLM sees their scores and counts, never their rows.
6. **Intermediate files** may be provided to the organisers. The email does not change the data licence, which forbids
   redistributing the data or publishing derived data that would rebuild it, and our repository will be public. So
   files of intermediate steps go to the organisers directly, never into the repository, unless they say otherwise.
   Issue #14 (26 Oct) allows a small intermediate output in the repository with the script that regenerates it, as
   long as it carries no tracking, video or audio ("Settled without asking", below).
7. **Other datasets:** any we have the rights to use, with the licence checked and named in the technical document,
   and every labelled one disclosed (what, how, by whom, how much), as SUBMISSION.md asks. The DFL/IDSSE data is
   CC BY 4.0 and PathCRF's checkpoint MPL-2.0 (dropped). Metrica's sample data has no licence file; its README asks
   users to acknowledge the source, which we do (question 3).

## Where we stand on 3 Nov

The audit of `main.toml` in [technical_document.md](technical_document.md) section 5, read against the email (A21 in
[results.md](results.md)). The two Smart Tagging rows describe `main.toml` as it was before R029, which took the source
out the same day:

| part of `main.toml` or of how it was chosen | how it was set | under the email |
|---|---|---|
| `controlled_touch` radius, speed, min_frames and hold; `hmm` 4/32/35/25; `shape_possession` weight and scope | the DFL benchmark, by rules declared before the runs (R010, R011, R013, R024) | external data (but see the degradation row) |
| `controlled_touch` confidence 0.93 | its agreement with DFL's flag (R013) | external data |
| the `dead_ball_model` and `shape_possession` models | trained on DFL's flags, on degraded tracking (R009, R011) | external data (but see the degradation row) |
| `shift_edges` +1 / −2 / 0 | the pipeline's own lag on DFL (R012); the public window's g is retired | external data |
| `D` before each half's kick-off | the organisers' documented kick-off frames | a documented fact |
| **`smart_tagging` c_team 0.76** | the tags' agreement with our own touch-rule pseudo-labels on BAR-ATH (R017) | **conflicted:** a parameter calibrated on the provided game, not on external data, and no external data has Smart Tagging. **Taken out of `main.toml` by R029** |
| **`smart_tagging` weight 0.2, spread `frame`** | Tian's pick of two declared options, made after seeing label-free aggregates of our own output on BAR-ATH (R017) | **doubtful:** not fitted to a number, but chosen while looking at this match's output, and never tested on external data. **Gone with the source (R029)** |
| **the benchmark's `tracking` v3 degradation**, the hazard model included | fitted once on label-free statistics of BAR-ATH's tracking (A07 to A09), then frozen | **open:** it is not the algorithm, and no ground truth or video enters it, but every external choice and both models were made on data it shaped. Our reading is that this is allowed, because the training and the tuning use external data and their labels. Ask (question 1) |
| R011's transfer gate and R013's label-free checks on BAR-ATH | pass or fail | automatic checks |
| `nearest_player` 0.78, `ball_presence` 0.2 / 0.585, from the public window (A01) | not in `main.toml`; they sat in R009's candidate configs, as the technical document discloses | not changed by the email (already ruled out on 23 Oct) |

**Before the upload.** *Done:* `smart_tagging` is out of `main.toml` (R029, 3 Nov, Tian: "it violates the new rules").
The file now has sha256 `762f9073…`. R024's `e6496a9b…` is not to be uploaded. Every remaining constant of `main.toml`
is set on external data, by a fixed rule or from a documented fact. The one question that could still move it is
question 1: if the answer is no, every learned part and every chosen constant has to be redone on a degradation that is
not fitted on our match. That is a large job, so ask early.

## What it rules out among the proposals

- **P14:** a ball detector trained or fine-tuned on our video is out, including one trained with the ATD's own ball as
  pseudo-labels. A detector trained on other data and run on our video (test B, zero-shot) stays allowed, because it
  uses the video "as input".
- **P22(c)**, adapting the shape model to our two teams without labels, is out. P22(a), P22(b) and P23 F would be out
  too; they were already dropped (R020, R027).
- **P21's repair from jersey colours:** clusters or a contrastive model fitted on our frames are out. A colour model
  trained on other games would be allowed. P21 is dropped anyway (A16).
- **P15:** the broadcast is allowed as input. If it returns, its alignment to the tactical feed should be computed by
  code at run time, not kept as two constants measured on this match (A20).
- **P10:** a pretrained audio tagger run on our audio is allowed; its thresholds come from elsewhere. P10 is dropped.
- **P02:** a person may watch the truth's changes to understand what counts as possession, never our output (point 4).
- **P11, P13, P16:** unchanged; they use external data (P16: question 3).

## What the organisers' GitHub issues settled (read 4 Nov 2026)

All 31 issues and pull requests on the organisers' repository were read on 4 Nov ([A22](results.md)). The organisers
answered on GitHub until 26 Oct; from 3 Nov every question gets the same reply, to write to the email address, and the
issues are closed. What the answers settle, beyond what RULES.md and the email say:

| issue | asked | the organisers' answer | for us |
|---|---|---|---|
| #2 (22 Oct) | a ball detector trained on the match's video with the ATD as labels; a GPU to reproduce | not allowed: no labelling of the provided dataset, even automatic, to train a model; weights are fine if trained on other data with the source stated; runtime is not a requirement and a GPU is fine | P14 may run only weights trained elsewhere (already here); our run is CPU, 17 s |
| #8 (22 Oct) | how much hand labelling is "small" | none on BAR-ATH for any purpose; unlimited on other matches if accessible and documented | compliant: our labels are DFL's and Metrica's own |
| #10 (23 Oct) | own hand labels used only to choose parameter values | not allowed: it fits the match; annotate another match, apply the values unchanged, share the video and the annotations | compliant (A21) |
| #11 (26 Oct) | must reproduction be byte-identical | training need not reproduce; the CSV must, byte for byte: commit the weights, infer on the CPU, pin the dependencies, fix the seeds | compliant: JSON models, CPU, pins; R029's file still has to be regenerated from a fresh clone |
| #14 (26 Oct) | committing a stage's small intermediate output; watching clips to understand why the ball goes missing, then rules with thresholds from other matches | yes to both; not allowed: labels, corrections, or rules and constants tied to specific frames | recorded below ("Settled without asking"); question 4 asks whether the email's narrower wording changes the second part |
| #15 (26 Oct) | is look-ahead allowed | yes: the task is post-match; a causal approach is scored the same | the Viterbi decode over a whole half is fine |
| #16 (26 Oct) | training a team-shape prior on the IDSSE dataset | fully permissible: other games, open, inspectable; cite CC BY 4.0; choose every hyperparameter on IDSSE or other games, never on the 5 public minutes | the organisers bless the dataset P01, P05 and P07 rest on |
| #17 (28 Oct, 3 teams) | a won header without control is a 1.32 s possession in the public truth; how does the owner behave on deflections and loose balls, is there a minimum duration | **no answer** (closed 3 Nov with the boilerplate) | question 6 below |
| #31 (3 Nov, open) | does `D` start at the infringement, the whistle or the ball crossing the line; does it end at the restart's first touch | **no answer** | question 5 below; what can be said is in A22 |
| #20, #22, #25, #26, #27, #28, #29 (29 Oct to 1 Nov) | validation-only hand labels; a rented cloud GPU for inference; GPU reproducibility; checking output against the video and automatic checks; a second match with tags and labels; a second match from the same pipeline; a time limit | the boilerplate only; the email of 3 Nov answers #20 (no hand labels for any purpose) and #26 (no checking against the video; automatic checks are fine) | nothing to do; a second match, if ever released, would be the lawful data for Smart Tagging's confidence and for the degradation |

## Questions for the organisers

**How to ask: by email, to `hackathon.talentarena@mobileworldcapital.com`.** RULES.md says to open an issue, but on
3 Nov the organisers closed the open issues on their repository and asked that every question be sent to that address
(their replies on issues #25 and #26). Email is private, so other teams do not see our approach.

**Sent 3 Nov 2026** by Tian as team Un Dia de Partit (questions 1 to 3 in one email, question 4 in a follow-up the same
day). No answer by 4 Nov.

1. **The benchmark's degradation.** We train and choose everything on the seven DFL/IDSSE matches (CC BY 4.0) and
   their own labels, after degrading their tracking to look like ours. The degradation uses the provided tracking
   directly: its own missing-ball pattern (which frames of each half have no ball) is copied onto the external matches
   (`gaps`, and `tracking` on top of it), and the players are cut into tracklets over 46 slots, with wrong-team slots,
   missing speeds and smoothing, at rates measured on it (A07 to A09). Everything was measured once and frozen, and
   none of it involves ground truth, video or a hand label. Is that use of the provided tracking allowed, or must the
   degradation be designed without measuring the provided game? If the answer is no, every learned part and every
   chosen constant has to be redone.
2. **Our repository's visibility.** It is private while we work and will be made public before the deadline, in time
   for the form. Is that fine?
3. **Metrica's sample data** (github.com/metrica-sports/sample-data) has no licence file. We used the event files of
   its three games once, for R023's measurement, with acknowledgement and without redistributing them. Is that fine?
4. **Watching the video to understand the tracking.** Issue #14 (26 Oct) allowed watching clips to understand why the
   ball goes missing, and turning general observations into rules whose constants come from other matches. The
   clarification allows inspecting the video "to better understand what counts as possession". May team members still
   watch the video to understand how the tracking fails (general observations only, no frame-level record, nothing
   compared with our output)?

**Drafted 4 Nov 2026, to send** (the two public issues the organisers left unanswered, A22):

5. **When `D` starts and ends** (issue #31, 3 Nov, open and unanswered). The ground truth is `D` where the corrected
   tracking marks the ball dead. Is that flag set when the ball wholly crosses the line, at the moment of an
   infringement, or at the referee's whistle or signal? Does it end at the restart kick, when a throw-in enters the
   field, or at the first touch after the restart? Our measurement on the public window's four stoppages (A11) cannot
   tell the three apart.
6. **Short possessions without control** (issue #17, 28 Oct, asked by three teams, closed unanswered). The public truth
   labels a won header without further control as a possession of about 1.3 s. Does the corrected tracking's owner
   switch at every touch, so that the full-match reference also counts such touches as possessions, and is there a
   minimum duration or number of touches before it switches?

**Settled without asking:**
- *Smart Tagging's confidence measured on our match* (the old question 2): no longer needed, since R029 took the source
  out of `main.toml`. Ask only if the tags are ever wanted back.
- *Files of intermediate steps* (the old question 3): answered on issue #14 (26 Oct), before the email. A small
  intermediate output may be committed together with the script, model version and settings that regenerate it, so
  that the step from it to the CSV is exactly reproducible. That issue's output held event times and scores only, no
  tracking, video or audio. We have no such file today.

When an answer comes, record it here with its date and update the audit above.
