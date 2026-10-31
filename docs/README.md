# Team docs

Our working notes for the Phase 1 entry. Read [proposals.md](proposals.md) before starting something new and
[traps.md](traps.md) before debugging something strange.

| file | holds | update it when |
|---|---|---|
| [proposals.md](proposals.md) | ideas to try, prioritised, with owner and status | you have an idea, start one, or finish one |
| [results.md](results.md) | every experiment and analysis with its numbers, commit and command; the submission log | you ran something worth remembering — failures included |
| [traps.md](traps.md) | pitfalls, why they happen, what to do instead | something cost you more than half an hour |
| [technical_document.md](technical_document.md) | the full record behind the required technical document | a decision lands in the pipeline |
| [technical_document_submission.md](technical_document_submission.md) | the technical document at its submitted length: the file to upload | with the full record, when a decision changes it |

The pipeline is documented next to its code: [pipeline/README.md](../pipeline/README.md) has the overview and the
diagram, and each stage folder has its own README.

[site/index.html](../site/index.html) is the visual overview: the pipeline as a clickable diagram, its history and
the work left, in one page that opens from disk (how to update it: [site/README.md](../site/README.md)).

IDs: proposals `P01`, experiments `R001`, analyses `A01`, traps `T01`. Never reuse or renumber them, and
cross-reference freely (`R004 tests P05, hit T03`).

**This folder is public, and Claude reads it.** Scores, counts, frame numbers and descriptions of what you saw are
fine. Raw tracking, tagging or homography rows, video stills, or anything else copied out of `data/` are not — see
the hard rules in [CLAUDE.md](../CLAUDE.md).
