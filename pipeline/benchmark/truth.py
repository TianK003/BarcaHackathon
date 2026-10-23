"""The benchmark's truth: the scorer's H/A/D/X codes from a DFL match's own flags, and what to read off them.

    truth_codes(match)           H/A/D/X per row of a DflMatch, the way the organisers' ground truth is built
    summary(codes, graded, fps)  a truth the way coaches read it: how many possessions, how long, how many stoppages
    concatenate(parts)           several matches as one array, so that a pooled value is one summary and not an average
    line_timing(match, codes)    how long before DFL's alive -> dead flip the ball was first seen beyond a line
    nearest_team(match)          the team of the player nearest the ball, brute force, in metres
    nearest_agreement(...)       how often that team is the truth's
    lagged, lag_spell_f1         the truth moved k frames later, and its spellF1 against itself: the cost of being late

Why the truth is built this way
-------------------------------
Our ground truth is derived mechanically from corrected tracking: D where it marks the ball dead, otherwise the team it
names, and dead-ball ownership is never asked for (CLAUDE.md). DFL's positions carry the same two flags on every frame,
BallStatus (alive or dead) and BallPossession (which team), so the same rule gives a truth for a match that is not
ours: D where the ball is dead, and where it is alive the team in possession. DFL sets BallPossession on dead frames
too, which this rule ignores, as the organisers' does. The codes are the scorer's
own (possession_metrics.H, A, D, X), so a truth goes straight into pm.evaluate, and outside the two graded halves it
says X, so that the scorer's valid runs are exactly the halves (pm.valid_runs(truth) == the halves, as for BAR-ATH).

The flags are complete in all seven downloaded matches (docs/results.md, A05: no frame lacks a status or a possession),
so the carrying rules below are guards that a clean match never uses, and the report says how often it did.

Carrying, inside each graded half only, never across half time
--------------------------------------------------------------
* An alive frame whose possession is unknown takes the last known possession of the half, counting every frame
  (DFL sets it on dead frames too) and, at the head of the half, the next one. Counted as carried_possession_frames.
* A frame whose status is unknown takes the label of the previous frame of the half, or at its head the next one.
  Counted as carried_status_frames.
* A half whose status is never known, or which is alive with no possession anywhere, has nothing to carry: ValueError.
* A code that is none of DFL's (0, 1, -1) is refused rather than read as a team.

Units
-----
A truth is an int8 array with one entry per row of the match, as long as the match's timeline (the 25 separator rows
between the halves are X). `graded` is a tuple of (half, start, stop) with half-open rows, as in DflMatch.graded; for a
truth that is not from a DflMatch (the public GT) it is ((1, start, stop), ...) from pm.valid_runs.
"""

import numpy as np

import possession_metrics as pm

# Params
ALIVE, DEAD, ABSENT = 1, 0, -1          # DflMatch.ball_status; ABSENT is also "no possession"
HOME, AWAY = 0, 1                       # DflMatch.ball_possession and player_team
STATUS_CODES = (ABSENT, DEAD, ALIVE)
POSSESSION_CODES = (ABSENT, HOME, AWAY)
SHORT_RUN_FRAMES = (2, 12)              # summary counts the possessions of at most this many frames: 0.08 s and 0.48 s
UNDER_S = 5                             # share_under_5s: possessions shorter than this many seconds
LINE_WINDOW_S = 10                      # line_timing looks this far back from a flip to dead
CHUNK = 20_000                          # rows per block of nearest_team (about 13 MB per array for 40 players)


# -- the codes --------------------------------------------------------------------------------------------------------

def truth_codes(match):
    """(codes, report): int8 pm codes per row of `match`, and what had to be carried.

    X outside the graded halves; inside one, D where the ball is dead, H or A (the team in possession) where it is
    alive. See the module docstring for the carrying rules. The report counts carried_possession_frames (alive frames
    whose team was carried), carried_status_frames (frames whose label was carried) and graded_frames.
    """
    codes = np.full(match.n_frames, pm.X, dtype=np.int8)
    carried_possession = carried_status = graded_frames = 0
    for half, start, stop in match.graded:
        status = np.asarray(match.ball_status[start:stop])
        possession = np.asarray(match.ball_possession[start:stop])
        _refuse_odd_codes(match, half, "ball_status", status, STATUS_CODES)
        _refuse_odd_codes(match, half, "ball_possession", possession, POSSESSION_CODES)
        known_status = status != ABSENT
        known_team = possession != ABSENT
        alive = status == ALIVE
        if not known_status.any():
            raise ValueError("{}: no frame of half {} has a ball status, so there is nothing to label it from; is the "
                             "positions file complete?".format(match.match_id, half))
        if alive.any() and not known_team.any():
            raise ValueError("{}: the ball is alive in half {} and no frame of it has a possession, so no team can be "
                             "named; is the positions file complete?".format(match.match_id, half))

        label = np.full(stop - start, pm.D, dtype=np.int8)
        if alive.any():
            team = possession[_latest_known(known_team)]
            label[alive] = np.where(team[alive] == HOME, pm.H, pm.A)
        codes[start:stop] = label[_latest_known(known_status)]
        carried_possession += int((alive & ~known_team).sum())
        carried_status += int((~known_status).sum())
        graded_frames += stop - start
    return codes, {"carried_possession_frames": carried_possession, "carried_status_frames": carried_status,
                   "graded_frames": graded_frames}


def _refuse_odd_codes(match, half, name, values, allowed):
    """ValueError if `values` holds a code that is not one of DFL's `allowed` ones (it would be read as a team)."""
    odd = sorted(set(np.unique(values).tolist()) - set(allowed))
    if odd:
        raise ValueError("{}: {} holds {} in half {}, and DFL's codes are {}; the cache may be damaged, rebuild it "
                         "with dfl.load_match(match_id, rebuild=True)".format(match.match_id, name, odd, half,
                                                                              list(allowed)))


def _latest_known(known):
    """For each position, the index of the latest known one at or before it; before the first, the first known one.

    The scorer's own carrying rule for an abstention (pm.fill_abstentions): forward, then back-filled at the head.
    """
    latest = np.maximum.accumulate(np.where(known, np.arange(len(known)), -1))
    latest[latest < 0] = int(np.argmax(known))
    return latest


# -- what a truth looks like ------------------------------------------------------------------------------------------

def summary(codes, graded, fps):
    """A truth the way coaches read possession: counts and lengths of possessions and stoppages, per half, pooled.

    `graded` is ((half, start, stop), ...). Keys and definitions:
      minutes             graded frames / fps / 60
      alive_share         (H + A) / graded frames
      possessions         maximal runs of H or of A inside a half (pm.possession_spells): H H D D H H is two
      per_alive_minute    possessions / minutes of H + A
      median_s, mean_s    possession lengths in seconds
      share_under_5s      the share of possessions shorter than 5 s
      runs_le_2_frames, runs_le_12_frames   how many possessions last at most 2, and at most 12, frames (0.08 s, 0.48 s)
      stoppages           maximal runs of D inside a half (pm.segments), stoppage_median_s their median length
      home_share          H / (H + A)
    A statistic of nothing (no possession, no alive frame) is NaN, and a count is 0. An X inside a half is neither alive
    nor dead; a truth has none.
    """
    spans = [(start, stop) for _, start, stop in graded]
    frames = sum(stop - start for start, stop in spans)
    home = sum(int((codes[start:stop] == pm.H).sum()) for start, stop in spans)
    away = sum(int((codes[start:stop] == pm.A).sum()) for start, stop in spans)
    alive = home + away
    lengths = np.array([stop - start for start, stop, _ in pm.possession_spells(codes, spans)], dtype=float)
    stops = np.array([stop - start for start, stop, code in pm.segments(codes, spans) if code == pm.D], dtype=float)
    nan = float("nan")
    return {
        "minutes": frames / fps / 60,
        "alive_share": alive / frames if frames else nan,
        "possessions": int(lengths.size),
        "per_alive_minute": lengths.size / (alive / fps / 60) if alive else nan,
        "median_s": float(np.median(lengths)) / fps if lengths.size else nan,
        "mean_s": float(lengths.mean()) / fps if lengths.size else nan,
        "share_under_5s": float((lengths < UNDER_S * fps).mean()) if lengths.size else nan,
        "runs_le_2_frames": int((lengths <= SHORT_RUN_FRAMES[0]).sum()),
        "runs_le_12_frames": int((lengths <= SHORT_RUN_FRAMES[1]).sum()),
        "stoppages": int(stops.size),
        "stoppage_median_s": float(np.median(stops)) / fps if stops.size else nan,
        "home_share": home / alive if alive else nan,
    }


def concatenate(parts):
    """(codes, graded) of several truths laid end to end, one X row between them, the spans moved to match.

    `parts` is [(codes, graded), ...]. A pooled value is the summary of this array, never an average of per-match values
    (a median of medians is not the median, and a big match would weigh the same as a short one). The X row keeps
    the last half of one match and the first half of the next from becoming one valid run for the scorer.
    """
    if not parts:
        raise ValueError("nothing to concatenate: pass at least one (codes, graded)")
    pieces, graded, offset = [], [], 0
    for index, (codes, spans) in enumerate(parts):
        if index:
            pieces.append(np.array([pm.X], dtype=np.int8))
            offset += 1
        pieces.append(np.asarray(codes, dtype=np.int8))
        graded += [(half, start + offset, stop + offset) for half, start, stop in spans]
        offset += len(codes)
    return np.concatenate(pieces), tuple(graded)


def line_timing(match, codes, alive_only=False):
    """How DFL's flip to dead sits against the ball leaving the pitch, over every alive -> dead change of the truth.

    A change is a frame t inside a half with D, after an H or an A. The ball is beyond a line where its normalised
    centre has x < 0, x > 1, y < 0 or y > 1 (a frame without the ball is neither). The window of a change is the
    LINE_WINDOW_S seconds before it, [t - 10 fps, t), clipped to the start of the half. Reported:
      changes, with_beyond_line   how many changes, and how many have the ball beyond a line in their window
      share_beyond_line           their ratio
      median_s, p25_s, p75_s      over those changes, the seconds from the first frame in the window with the ball
                                  beyond a line to the change: how long the flag waited after the ball went out
      leads_s                     those seconds, change by change, for pool_line_timing
    A restart that keeps the ball inside the pitch (a foul, an offside) has no such frame, so it is in the share and
    not in the seconds. NaN where there is nothing to take a statistic of.

    The window can reach back into an earlier dead stretch (a throw-in taken less than 10 s before the next stop): its
    frames have the ball beyond a line too, and the "first" such frame is then the stretch's, which inflates the share
    and the upper quartile. alive_only=True looks only at the frames of the window that the truth calls alive, so that
    what is measured is how long the flag waited, with the ball already out, to turn dead.
    """
    x, y = match.ball_xy[:, 0], match.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        beyond = (x < 0) | (x > 1) | (y < 0) | (y > 1)
    if alive_only:
        beyond &= (codes == pm.H) | (codes == pm.A)
    window = LINE_WINDOW_S * match.fps
    changes, leads = 0, []
    for _, start, stop in match.graded:
        inside = codes[start:stop]
        flips = np.flatnonzero((inside[1:] == pm.D) & ((inside[:-1] == pm.H) | (inside[:-1] == pm.A))) + 1 + start
        for frame in flips.tolist():
            changes += 1
            first = max(start, frame - window)
            out = np.flatnonzero(beyond[first:frame])
            if out.size:
                leads.append((frame - (first + int(out[0]))) / match.fps)
    return _timing_report(changes, leads)


def pool_line_timing(results):
    """The line_timing of several matches as one: changes and leads added up, the statistics taken over them all.

    Never an average of per-match medians (a median of medians is not the median of the changes).
    """
    return _timing_report(sum(result["changes"] for result in results),
                          [lead for result in results for lead in result["leads_s"]])


def _timing_report(changes, leads):
    nan = float("nan")
    return {
        "changes": changes, "with_beyond_line": len(leads),
        "share_beyond_line": len(leads) / changes if changes else nan,
        "median_s": float(np.median(leads)) if leads else nan,
        "p25_s": float(np.percentile(leads, 25)) if leads else nan,
        "p75_s": float(np.percentile(leads, 75)) if leads else nan,
        "leads_s": tuple(leads)}


# -- the nearest team -------------------------------------------------------------------------------------------------

def nearest_team(match):
    """int8 per row: the team (0 home, 1 away) of the player nearest the ball, or -1 where there is no ball or nobody.

    Brute force and in metres: the normalised coordinates times the pitch, because the pitch is not square and a
    normalised distance would favour the cross-pitch direction. Only players who are on the pitch (not NaN) count; on a
    tie the lowest column wins. It is the simplest evidence for who has the ball, and A01 measured it on our match:
    right on 77.8% of alive, ball-visible frames of the public window.
    """
    scale = np.array(match.pitch, dtype=float)
    team = np.asarray(match.player_team, dtype=np.int8)
    nearest = np.full(match.n_frames, ABSENT, dtype=np.int8)
    for first in range(0, match.n_frames, CHUNK):
        rows = slice(first, min(first + CHUNK, match.n_frames))
        offset = (match.players_xy[rows] - match.ball_xy[rows][:, None, :]) * scale
        squared = (offset ** 2).sum(axis=2)
        squared[np.isnan(squared)] = np.inf                       # a player who is absent, or a ball that is
        column = squared.argmin(axis=1)                           # the first of equals: the lowest column
        found = np.isfinite(squared[np.arange(len(column)), column])
        block = nearest[rows]
        block[found] = team[column[found]]
    return nearest


def nearest_agreement(codes, nearest, spans):
    """(frames where the nearest player's team is the truth's, frames compared) over the half-open `spans`.

    Compared are the alive frames (H or A) that have a nearest team: H agrees with team 0, A with team 1.
    """
    hits = total = 0
    for start, stop in spans:
        code, team = codes[start:stop], nearest[start:stop]
        compared = ((code == pm.H) | (code == pm.A)) & (team >= 0)
        total += int(compared.sum())
        hits += int((compared & (team == np.where(code == pm.H, HOME, AWAY))).sum())
    return hits, total


# -- the cost of being late -------------------------------------------------------------------------------------------

def lagged(codes, k):
    """The truth moved k frames later inside each of its valid runs (the scorer's: pm.valid_runs), X before and between.

    The first k frames of a run are X, which the scorer fills with the first label it does know (pm.fill_abstentions),
    and the last k frames of the truth fall off the end. Nothing moves across the hole between the halves.
    """
    late = np.full_like(codes, pm.X)
    for start, stop in pm.valid_runs(codes):
        if start + k < stop:
            late[start + k:stop] = codes[start:stop - k]
    return late


def lag_spell_f1(codes, k):
    """spellF1 (the scorer's spell_f1_mean) of the truth against itself lagged k frames: what k frames of lag cost."""
    return pm.evaluate(codes, lagged(codes, k))["spell_f1_mean"]
