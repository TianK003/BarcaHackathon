# site/: the visual overview

`index.html` is a one-page, interactive overview of our Phase 1 entry, for the team: where the submission stands,
the pipeline as a clickable diagram, how it works, its history (a timeline, the benchmark score by entry, how old
each component is) and the work left before the deadline. It summarises the team docs; it does not replace them.

`rules.html` is its companion: the competition rules as the organisers wrote them (RULES.md is the authority), what
each means for us, our stricter team practices kept apart from the rules, a self-check, and the rules' history. It
says when it was last checked against the organisers' repo; re-check with
`git fetch upstream && git log main..upstream/main` and update its DATA when they change a rule.

`progress.html` is the third page: pooled benchmark spellF1 by entry as one chart (adopted entries filled and joined
by a step line, the running `main.toml`; candidates not adopted hollow), the feature ledger (every measured proposal,
the drops included), the public window as a correctness check only, and how to add a point. Its DATA holds `points`
(one per entry that scored `main.toml` on the benchmark), `ledger` (one row per measured proposal, always), `features`
(final statuses, which the cards count) and `publicWindow`. When a lane finishes, append its entry to `points` (only if
results.md gives a benchmark score) and to `ledger`, check the numbers against `docs/results.md`, and bump `asOf` and
`footer`. A dash in the ledger means results.md has no benchmark number; never invent one.

## Open it

Each of the three pages is one self-contained file (inline CSS, JavaScript and SVG) that makes no network
requests, so any of these
work:

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
   `params` or `retuned`, a proposal's `status`, a stage's `notBuilt`, the `cards`, the `deliverables`.
2. Check each number against its source: `docs/results.md` (scores, entries, the submission log) and
   `docs/proposals.md` (statuses, the "After …" recommendations). The docs are the source of truth; this page only
   repeats them.
3. Bump `asOf`, `commit` and `footer`.
4. Open the page and click through the tabs; the browser console should stay empty.

Text fields understand a little markdown: `` `code` ``, `**bold**` and `[label](relative/link)`.

### Built and not built, on the Pipeline tab

The diagram marks what is built in green and what is not built yet in orange.

- **A stage's box** is green when its entry in `stages` has a `built` line (all seven do today) and orange, dashed,
  when it has none.
- **The chips inside a stage** are its `components` (for `configs`, the files in `configs`): filled green with ✓
  when in `main.toml`, outlined green with ○ when built but not in `main.toml`. Dropped components are left out of the
  box and keep their badge in the stage's details.
- **The orange chips** come from the stage's `notBuilt` list (`{ name, p }`, a short name and the open proposal it
  belongs to): the ideas that would add to the stage and are not built. When one is built, move it into `components`;
  when its proposal is dropped, delete it from `notBuilt`. The same idea may sit under two stages (P11 adds a source
  and a refiner).
- **In a stage's details,** built components have a green edge and open proposals (`idea`, `partly`) and the stage's
  own planned items an orange one.

## Never

- **No data rows.** The repository will be public and this page is read by people and by Claude. It holds
  aggregates and descriptions only: scores, counts, shares, dates, entry IDs. Never paste tracking, tagging or
  homography rows, labels of frames of our match, or anything else copied out of `data/` (CLAUDE.md, hard rules).
- **No invented numbers.** If a fact is not in the docs, leave it out.
- **No external resources**: no CDN, web font or analytics. The page must keep working offline.
