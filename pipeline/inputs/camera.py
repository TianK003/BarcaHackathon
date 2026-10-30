"""Where the tactical camera looks: the pitch point under the image centre, from the per-frame homography.

The homography maps normalised pitch coordinates (0-1) to pixels, so the point under a pixel is found by inverting it,
not by applying it: solve H @ (u, v, 1) ~ (cx, cy, 1) for (u, v) and scale u by the pitch's length. Only the along-pitch
coordinate is returned, the one a follower camera pans in; the across-pitch tilt is small and no feature asked for it.

The result is a property of the video, not of the ball or the players, so it holds no truth (P09). It is match data, like
the homography it comes from: print aggregates of it, never rows.
"""

import numpy as np

# Params
CENTRE = (960.0, 540.0)     # the image centre of the 1920 x 1080 tactical video, in pixels


def aim_x(matrices, valid, pitch_length):
    """(n,) metres along the pitch of the point under the image centre; NaN where `valid` is False or H is singular.

    `matrices` is the (n, 3, 3) homography (normalised pitch -> pixels) and `valid` the (n,) bool that says which rows
    are usable. A solution at infinity (the centre pixel on the horizon) or a non-finite one is NaN too.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or valid.shape != matrices.shape[:1]:
        raise ValueError("matrices must be (n, 3, 3) and valid (n,), got {} and {}".format(matrices.shape, valid.shape))
    out = np.full(len(matrices), np.nan)
    rows = np.flatnonzero(valid & np.isfinite(matrices).all(axis=(1, 2)))
    if not len(rows):
        return out
    # H @ (u, v, 1) = w * (cx, cy, 1), w unknown: (H0 - cx H2) p = 0 and (H1 - cy H2) p = 0 are two linear equations
    # in (u, v) with p = (u, v, 1), so the stacked 2 x 2 system is solved per row.
    h = matrices[rows]
    a = np.stack([h[:, 0, :2] - CENTRE[0] * h[:, 2, :2], h[:, 1, :2] - CENTRE[1] * h[:, 2, :2]], axis=1)
    b = -np.stack([h[:, 0, 2] - CENTRE[0] * h[:, 2, 2], h[:, 1, 2] - CENTRE[1] * h[:, 2, 2]], axis=1)
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    ok = np.abs(det) > 1e-12 * np.maximum(1.0, np.abs(a).max(axis=(1, 2)) ** 2)
    if ok.any():
        uv = np.linalg.solve(a[ok], b[ok][:, :, None])[:, :, 0]
        out[rows[ok]] = np.where(np.isfinite(uv[:, 0]), uv[:, 0] * pitch_length, np.nan)
    return out
