# Rules: the organisers' clarification of November 2026

**This file is the team's statement of the rules, since 3 Nov 2026.** The organisers emailed every participant a
"Clarification of the rules" (verbatim below). They say it "take[s] precedence over any other information you may have
received previously, whether in the documentation, in earlier emails or in answers to individual questions". So where it
differs from RULES.md, FAQ.md, EVALUATION.md, SUBMISSION.md, OVERVIEW.md or the clarifications of 22 and 23 Oct
(upstream `c073f29`, `f8f2629`), the email wins, and it also wins over anything our own docs said before 3 Nov. Where
the email says nothing, those documents still hold. That covers rule 3 (the repository regenerates the same file), the
data licence accepted at enrolment, the deliverables and the conditions on what we may use (the organisers can access
it, the technical document explains it).

On 5 Nov the organisers answered our questions 1 to 4 with this clarification again, and one new sentence: "anything
that is not explicitly allowed is not permitted" ([The organisers' reply of 5 Nov](#the-organisers-reply-of-5-nov-2026)).

Upstream has no commit carrying the email yet: on 3 Nov `git fetch upstream` still ended at `f8f2629`, and on 4 and 9 Nov
it still does. When one lands, compare it with this file and update both. On 9 Nov the organisers' repository had its
issues and pull requests switched off (the GitHub API reports `has_issues: false`, and its pull-request list returns 404),
so the answers given there ([below](#what-the-organisers-github-issues-settled-read-4-nov-2026)) can no longer be read
on GitHub; this file is our record of them.

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
>
> If you have any questions, don't hesitate to write to us.
>
> We wish you the best of luck!

## The organisers' reply of 5 Nov 2026

Questions 1 to 4 ([below](#questions-for-the-organisers)) went by email on 3 Nov. The Talent Arena team replied on
5 Nov 2026 (recorded here on 9 Nov). The reply answers none of the four by name. It says that "all the information we
can share has already been provided in the clarification email, the challenge details and the terms and conditions",
repeats the clarification above word for word ("What is not allowed", "What is allowed", "Training and tuning"), and
ends with two sentences that are new:

> Each team is responsible for making sure that its particular project, tool, platform or dataset complies with the
> competition rules. As a reference, anything that is not explicitly allowed is not permitted.

**What it changes.** The second sentence is a rule for reading the others, and it is stricter than our own was. Until
now we took the stricter reading where the text left room, *and asked before relying on a looser one*. The organisers
have now said two things. Asking will not get us a looser reading: they answer with the clarification. And whatever
the clarification, the challenge details and the T&C do not allow explicitly is not permitted. So an open case now
closes the strict way: if no text allows it by name, we treat it as forbidden. The sentence came in a reply to us, not
in a published rule, but it contradicts nothing in the clarification, and we follow it.

**Each question, read under it:**

| question | does a text allow it explicitly? | our reading since 5 Nov | what follows |
|---|---|---|---|
| 1. the benchmark's degradation, fitted on label-free statistics of our tracking | no (below) | **not permitted** | every learned part and every chosen constant rests on it. Redo them on a degradation that measures nothing of our match (P27), or keep them and carry the risk. **Decided 9 Nov (Tian): redo, P27** |
| 2. the repository private until shortly before the deadline | the rules ask for a public repository URL in the form and say nothing about when it becomes public | **fine** | make it public before the form is submitted, after checking the whole git history, not only the tip, for anything under `data/` or frame-level. A deliberate step, Tian's |
| 3. Metrica's sample data, used once for R023 | no: "Using any dataset you have the rights to use. Please check the license". A repository without a licence grants no rights; its README asks for acknowledgement, which is not permission. That Metrica Sports, one of the organisers, publishes it does not license it either | **not permitted** | no further use: P16's timing part and P26 are on hold and `metrica.py` is not run again; the technical document lists the data as used once and withdrawn; R024's objective, which R023 set, stays (decided 9 Nov, below) |
| 4. watching the video to understand how the tracking fails | no: watching and inspecting are allowed "to better understand what counts as possession", nothing more. Issue #14's wider answer (26 Oct) is superseded, since the clarification takes precedence over "answers to individual questions" | **not permitted** | no change: our practice since 3 Nov ("How we read it", point 4) |

**Question 1 in detail.** Three items of the clarification come near it, and none allows it by name.
- "Running automatic checks using only the provided data" covers a check that passes or fails. It does not cover a
  measurement that is written into training data.
- "Using as input ... the automatic tracking data" covers what the algorithm reads at run time. The degradation is not
  the algorithm: it makes the algorithm's training and validation data.
- "If you need to train (or retrain), you MUST use only external data" is the line it crosses. `gaps` transplants our
  own missing-ball masks onto the DFL matches byte for byte, and `tracking` v3 adds tracklet lengths, wrong-team and
  NaN-speed shares and a rescale, all measured on our tracking (A07 to A09). So both models were trained on data that
  carries measurements of the provided game (R009, R011), and so was every constant chosen after them (R010 to R024).

By T54's own test (does a number go from BAR-ATH into a config, a model file, a prior or a threshold?), the
degradation's constants are numbers from BAR-ATH in the code that makes the training data. We read it as allowed
because it is not the algorithm and sees no ground truth or video. That was the looser reading, and the reply takes
away its footing.

If the team follows the reading, everything below has to be redone on a degradation that measures nothing of our match
([P27](proposals.md)):
- retrain `dead_ball_model` and `shape_possession` on it;
- re-run the choices of R010, R011, R013 and R024 by their declared rules;
- regenerate the file.

The fidelity gates compare a profile with our match. They may still run as a check that passes or fails, but no
profile may be changed to pass them (T01). Nor may what we measured on BAR-ATH ourselves (a median gap of 10.7 s, the
tracklet lengths, the wrong-team share) become the new profile's settings: that would be the same fit, made by hand.
The figures that the organisers publish in DATA.md are another matter (41% of the graded frames without a ball, a mean
of 20.5 tracked objects, 46 slots): they are part of the task statement, like the kick-off frames, and P27 quotes
them ([the spec](superpowers/specs/2026-11-09-p27-degradation-design.md); the advisor's ruling of 9 Nov).

**Question 3 in detail.** R023 fetched the four event files once (31 Oct) and measured the grain of possession
derived from them. Its declared outcome, "both", made the unweighted mean of the raw and coarse truths the objective of
every choice after it. R024 re-chose R013's grids under that objective, which moved the touch hold from 0.60 to 0.55
(+0.0052 on the mean). Without R023, the grain of our truth is unknown again, which is where A06 left it: rank under
both truths and treat a disagreement as a warning. There are two ways to go:
- keep R024's hold: the mean of the two truths is also the neutral way to rank when neither is known to be right;
- or put `main.toml` back to R013's 0.60.

**Decided 9 Nov (Tian): keep the average.** The mean of the two truths stays the objective, on A06's ground: the
grain of our truth is unknown, so neither truth is preferred. P27 re-chooses every constant under it.

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
| Anything the texts do not name | ask before building on it (RULES.md, Questions) | **not permitted**: "anything that is not explicitly allowed is not permitted" (the reply to our questions, 5 Nov) |

## How we read it

Where the text leaves room we take the stricter reading. Since 5 Nov that is the organisers' own rule: "anything that
is not explicitly allowed is not permitted", and a question gets the clarification back
([their reply](#the-organisers-reply-of-5-nov-2026)). So an open case is not left open while we wait for an answer:
it is read strictly from the start.

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
   CC BY 4.0 and PathCRF's checkpoint MPL-2.0 (dropped). Metrica's sample data has no licence (GitHub reports none,
   checked again on 9 Nov). Its README asks users to acknowledge the source, which is a request and not a grant of
   rights, so since the reply of 5 Nov we do not use it (question 3; R023 used it once, and the technical document
   says so).

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
| **the benchmark's `tracking` v3 degradation**, the hazard model included | fitted once on label-free statistics of BAR-ATH's tracking (A07 to A09), then frozen | **not permitted by our reading since 5 Nov** (until then open: it is not the algorithm, and no ground truth or video enters it, so we read it as allowed and asked, question 1). The organisers' reply allows nothing that is not allowed explicitly, and nothing allows it; both models and every external choice were made on data it shaped. **Decided 9 Nov (Tian): redo on P27** |
| **the objective of R024's choice** (the unweighted mean of the raw and coarse truths), which moved the touch hold to 0.55 | R023's grain, measured on Metrica's sample games | **doubtful since 5 Nov:** the sample data has no licence, so by our reading we do not have the rights to use it (question 3). **Decided 9 Nov (Tian): the objective stays**, on A06 alone (the grain unknown, both truths ranked); P27 re-chooses the hold under it |
| R011's transfer gate and R013's label-free checks on BAR-ATH | pass or fail | automatic checks |
| `nearest_player` 0.78, `ball_presence` 0.2 / 0.585, from the public window (A01) | not in `main.toml`; they sat in R009's candidate configs, as the technical document discloses | not changed by the email (already ruled out on 23 Oct) |

**Before the upload.** *Done:* `smart_tagging` is out of `main.toml` (R029, 3 Nov, Tian: "it violates the new rules").
The file now has sha256 `762f9073…`. R024's `e6496a9b…` is not to be uploaded. Every remaining constant of `main.toml`
is set on external data, by a fixed rule or from a documented fact. The one question that could still move it is
question 1: if the answer is no, every learned part and every chosen constant has to be redone on a degradation that is
not fitted on our match. That is a large job, so ask early.

**After the reply of 5 Nov (recorded 9 Nov).** The reply did not answer question 1. It added "anything that is not
explicitly allowed is not permitted", and nothing allows the degradation explicitly. So by our reading the degradation
and R024's objective (the two rows marked "since 5 Nov" above) no longer stand on the rules as written. `main.toml` and its
file (`762f9073…`) are unchanged until the team decides. There are two ways to go, with eight days to the deadline:
- **redo** on a degradation that measures nothing of our match ([P27](proposals.md));
- **keep** and disclose the degradation and R023 in the technical document as our reading's open risk.

The redo is the one that follows the organisers' sentence. Most of its work is re-running scripts that exist (the
two models' training, `run.py` and `choose.py` on each grid's declared rule). The new part is the profile itself.

**Decided 9 Nov 2026 (Tian):** (1) redo the degradation: P27, started the same evening; (2) keep the average: the
mean of the two truths stays the objective; (3) questions 5 and 6 are not sent.

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
- **P11, P13:** unchanged; they use external data, each source's licence checked first.
- **P16's timing part and P26** (5 Nov): on hold. Metrica's sample data has no licence, so by our reading we do not
  have the rights to use it (question 3).
- **P01's degradation** (5 Nov): by our reading not permitted, since it is fitted on our tracking (question 1). P27 is
  the redo (decided 9 Nov).

## What the organisers' GitHub issues settled (read 4 Nov 2026)

All 31 issues and pull requests on the organisers' repository were read on 4 Nov ([A22](results.md)). The organisers
answered on GitHub until 26 Oct; from 3 Nov every question gets the same reply, to write to the email address, and the
issues are closed. By 9 Nov issues and pull requests are switched off on the repository, so this table is our only
record of the answers. The clarification takes precedence over "answers to individual questions", so an answer below
holds only where the clarification agrees with it. What the answers settle, beyond what RULES.md and the email say:

| issue | asked | the organisers' answer | for us |
|---|---|---|---|
| #2 (22 Oct) | a ball detector trained on the match's video with the ATD as labels; a GPU to reproduce | not allowed: no labelling of the provided dataset, even automatic, to train a model; weights are fine if trained on other data with the source stated; runtime is not a requirement and a GPU is fine | P14 may run only weights trained elsewhere (already here); our run is CPU, 17 s |
| #8 (22 Oct) | how much hand labelling is "small" | none on BAR-ATH for any purpose; unlimited on other matches if accessible and documented | compliant: our labels are DFL's and Metrica's own |
| #10 (23 Oct) | own hand labels used only to choose parameter values | not allowed: it fits the match; annotate another match, apply the values unchanged, share the video and the annotations | compliant (A21) |
| #11 (26 Oct) | must reproduction be byte-identical | training need not reproduce; the CSV must, byte for byte: commit the weights, infer on the CPU, pin the dependencies, fix the seeds | compliant: JSON models, CPU, pins; R029's file still has to be regenerated from a fresh clone |
| #14 (26 Oct) | committing a stage's small intermediate output; watching clips to understand why the ball goes missing, then rules with thresholds from other matches | yes to both; not allowed: labels, corrections, or rules and constants tied to specific frames | the first part is recorded below ("Settled without asking"); the second is superseded: the clarification allows watching only "to better understand what counts as possession", and the reply of 5 Nov to question 4 added nothing |
| #15 (26 Oct) | is look-ahead allowed | yes: the task is post-match; a causal approach is scored the same | the Viterbi decode over a whole half is fine |
| #16 (26 Oct) | training a team-shape prior on the IDSSE dataset | fully permissible: other games, open, inspectable; cite CC BY 4.0; choose every hyperparameter on IDSSE or other games, never on the 5 public minutes | the organisers bless the dataset P01, P05 and P07 rest on |
| #17 (28 Oct, 3 teams) | a won header without control is a 1.32 s possession in the public truth; how does the owner behave on deflections and loose balls, is there a minimum duration | **no answer** (closed 3 Nov with the boilerplate) | question 6 below (drafted, not sent) |
| #31 (3 Nov) | does `D` start at the infringement, the whistle or the ball crossing the line; does it end at the restart's first touch | **no answer** (still open on 4 Nov; the issues are switched off by 9 Nov) | question 5 below (drafted, not sent); what can be said is in A22 |
| #20, #22, #25, #26, #27, #28, #29 (29 Oct to 1 Nov) | validation-only hand labels; a rented cloud GPU for inference; GPU reproducibility; checking output against the video and automatic checks; a second match with tags and labels; a second match from the same pipeline; a time limit | the boilerplate only; the email of 3 Nov answers #20 (no hand labels for any purpose) and #26 (no checking against the video; automatic checks are fine) | nothing to do; a second match, if ever released, would be the lawful data for Smart Tagging's confidence and for the degradation |

## Questions for the organisers

**How to ask: by email, to `hackathon.talentarena@mobileworldcapital.com`.** RULES.md says to open an issue, but on
3 Nov the organisers closed the open issues on their repository and asked that every question be sent to that address
(their replies on issues #25 and #26). Email is private, so other teams do not see our approach.

**Sent 3 Nov 2026** by Tian as team Un Dia de Partit (questions 1 to 3 in one email, question 4 in a follow-up the same
day). **Answered 5 Nov 2026 with the clarification only**, plus "anything that is not explicitly allowed is not
permitted"; what that means for each question is in [the reply](#the-organisers-reply-of-5-nov-2026) above. The
questions as sent (the email put them in fuller words):

1. **The benchmark's degradation.** We train and choose everything on the seven DFL/IDSSE matches (CC BY 4.0) and
   their own labels, after degrading their tracking to look like ours. The degradation uses the provided tracking
   directly: its own missing-ball pattern (which frames of each half have no ball) is copied onto the external matches
   (`gaps`, and `tracking` on top of it), and the players are cut into tracklets over 46 slots, with wrong-team slots,
   missing speeds and smoothing, at rates measured on it (A07 to A09). Everything was measured once and frozen, and
   none of it involves ground truth, video or a hand label. Is that use of the provided tracking allowed, or must the
   degradation be designed without measuring the provided game? If the answer is no, every learned part and every
   chosen constant has to be redone.
   *Reply (5 Nov): the clarification. Our reading: not permitted, since nothing allows it explicitly. Redo (P27) or
   keep: decided 9 Nov (Tian), redo.*
2. **Our repository's visibility.** It is private while we work and will be made public before the deadline, in time
   for the form. Is that fine?
   *Reply (5 Nov): the clarification. Our reading: fine. The form asks for a public URL and no rule says when the
   repository becomes public.*
3. **Metrica's sample data** (github.com/metrica-sports/sample-data) has no licence file. We used the event files of
   its three games once, for R023's measurement, with acknowledgement and without redistributing them. Is that fine?
   *Reply (5 Nov): the clarification, whose item is "any dataset you have the rights to use. Please check the license".
   Our reading: without a licence we do not have those rights. No further use; R024's objective stays (decided 9 Nov, Tian).*
4. **Watching the video to understand the tracking.** Issue #14 (26 Oct) allowed watching clips to understand why the
   ball goes missing, and turning general observations into rules whose constants come from other matches. The
   clarification allows inspecting the video "to better understand what counts as possession". May team members still
   watch the video to understand how the tracking fails (general observations only, no frame-level record, nothing
   compared with our output)?
   *Reply (5 Nov): the clarification. Our reading: no. Watching is only to understand what counts as possession, as
   since 3 Nov.*

**Drafted 4 Nov 2026, not sent** (the two public issues the organisers left unanswered, A22). The reply of 5 Nov says
that all the information they can share has been shared, so these would most likely get the clarification back.
**Decided 9 Nov (Tian): they are not sent.** So the docs treat both conventions as unknown.
A22's reading (dead when the ball leaves play or the referee stops it, alive at the restart) is a lead, and no constant
comes from the window.

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

When an answer comes, record it here with its date and update the audit above. The reply of 5 Nov is recorded in
[its own section](#the-organisers-reply-of-5-nov-2026).
