# site/: the visual overview

`index.html` is a one-page, interactive overview of our Phase 1 entry, for the team: where the submission stands,
the pipeline as a clickable diagram, how it works, its history (a timeline, the benchmark score by entry, how old
each component is) and the work left before the deadline. It summarises the team docs; it does not replace them.

## Open it

It is one self-contained file (inline CSS, JavaScript and SVG) that makes no network requests, so any of these work:

- double-click `site/index.html`;
- `start site/index.html` (Windows), `open site/index.html` (macOS), `xdg-open site/index.html` (Linux);
- any static server from the repo root, e.g. `python -m http.server`, then open `/site/`.

Links to repo files are relative (`../docs/results.md`), so they work from a checkout and on GitHub. Tabs have their
own addresses (`#overview`, `#pipeline`, `#how`, `#history`, `#work`, `#reference`), and a stage of the diagram can
be linked directly, e.g. `index.html#pipeline/decode`. The theme follows the system and can be switched by hand; the
choice is remembered in the browser. The day counts on the dates are computed from the viewer's own clock.

## Update it

Every fact lives in the `const DATA = { ... }` object at the top of the script. Edit DATA, never the markup or the
rendering code below it.

1. Change the facts that moved: a new entry in `timeline`, a point in `progression`, a component's `status`,
   `params` or `retuned`, a proposal's `status`, the `cards`, the `deliverables`.
2. Check each number against its source: `docs/results.md` (scores, entries, the submission log) and
   `docs/proposals.md` (statuses, the "After …" recommendations). The docs are the source of truth; this page only
   repeats them.
3. Bump `asOf`, `commit` and `footer`.
4. Open the page and click through the tabs; the browser console should stay empty.

Text fields understand a little markdown: `` `code` ``, `**bold**` and `[label](relative/link)`.

## Never

- **No data rows.** The repository will be public and this page is read by people and by Claude. It holds
  aggregates and descriptions only: scores, counts, shares, dates, entry IDs. Never paste tracking, tagging or
  homography rows, labels of frames of our match, or anything else copied out of `data/` (CLAUDE.md, hard rules).
- **No invented numbers.** If a fact is not in the docs, leave it out.
- **No external resources**: no CDN, web font or analytics. The page must keep working offline.
