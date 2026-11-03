# Rules: the organisers' clarification of November 2026

**This file is the team's statement of the rules, since 3 Nov 2026.** The organisers emailed every participant a
"Clarification of the rules" (verbatim below). They say it "take[s] precedence over any other information you may have
received previously, whether in the documentation, in earlier emails or in answers to individual questions". So where it
differs from RULES.md, FAQ.md, EVALUATION.md, SUBMISSION.md, OVERVIEW.md or the clarifications of 22 and 23 Oct
(upstream `c073f29`, `f8f2629`), the email wins, and it also wins over anything our own docs said before 3 Nov. Where
the email says nothing, those documents still hold. That covers rule 3 (the repository regenerates the same file), the
data licence accepted at enrolment, the deliverables and the conditions on what we may use (the organisers can access
it, the technical document explains it).

Upstream has no commit carrying the email yet: on 3 Nov `git fetch upstream` still ends at `f8f2629`. When one lands,
compare it with this file and update both.

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
   files of intermediate steps go to the organisers directly, never into the repository, unless they say otherwise
   (question 3).
7. **Other datasets:** any we have the rights to use, with the licence checked and named in the technical document,
   and every labelled one disclosed (what, how, by whom, how much), as SUBMISSION.md asks. The DFL/IDSSE data is
   CC BY 4.0 and PathCRF's checkpoint MPL-2.0 (dropped). Metrica's sample data has no licence file; its README asks
   users to acknowledge the source, which we do (question 4).

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
- **P11, P13, P16:** unchanged; they use external data (P16: question 4).

## Questions for the organisers

Not yet sent; sending them is the team's call. RULES.md asks for questions as issues on their public repository, but
an issue there shows our approach to every other team, so an email may be better.

1. Our validation degrades open tracking of other games (DFL/IDSSE, CC BY 4.0) to look like the provided tracking. A
   few label-free statistics of the provided tracking (how often and where the ball goes missing, how long the gaps
   last, how many player slots are present, the jitter) are measured once and frozen. Every parameter is then tuned,
   and our models trained, on the degraded external matches and their own labels. No ground truth and no video is
   involved. Is fitting that degradation to the provided tracking allowed?
2. *(No longer blocking: R029 took the source out. Ask only if we want the tags back.)* Smart Tagging exists only for
   the provided game. We measured how often its team votes agree with our own algorithm's labels on that game, with
   no ground truth and no person, and used the share as the votes' confidence. Is that allowed, or must such a
   parameter come from external data only?
3. "Providing files of intermediate steps": should they go into our public repository, given the data licence, or be
   sent to you?
4. Metrica's public sample data (github.com/metrica-sports/sample-data) has no licence file. May we use its event files
   for a measurement, with acknowledgement?

When an answer comes, record it here with its date and update the audit above.
