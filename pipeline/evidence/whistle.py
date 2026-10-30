"""P10 -- the referee's whistle, found in the tactical video's audio: the detector (the evidence source follows).

    features = inputs.audio_features                   # (n_frames, 2): tonal_db, band_db, per video frame (inputs/audio.py)
    w = whistle.detect(features, inputs.graded_mask)   # int64 first frames of the blasts

Why. A referee's whistle is a sustained tone at 2 to 4 kHz, well above most crowd noise, and play is dead in the moment
after it: either it has just been stopped, or it is about to be restarted by a kick that takes at least half a second to
follow the whistle. So where a blast is heard, the ball is likely dead, and in the gaps of our tracking that says
something the tracking cannot. The decision to use it is not made here: R018 (docs/results.md) declares a feasibility
gate and a bar, measured against our own decoded stoppages and no hand label, before the source is written.

The detector. A frame is "tonal" when both of its features reach their thresholds (tonal_db: the match's own 98th percentile, R018b), and a blast is a run of at least
MIN_FRAMES tonal graded frames; the detection time w is the run's first frame. The features are inputs/audio.py's, in dB
and rounded to 0.01 dB so that the comparison does not depend on the last bits of an FFT; a feature exactly at its
threshold counts (>=). The constants are argued from acoustics and fixed in R018's declaration, before any run of the
detector on the match; none is tuned to a score or to a stoppage.

Only the graded frames speak. A run is cut at the first frame that is not graded, so no blast crosses a block's edge
(halftime, the frames before the second half's kick-off), and a blast that starts before a block counts from the block's
first frame. Nothing is read across a gap.
"""

import numpy as np

# Params
TONAL_QUANTILE = 98.0  # R018b: a whistle fills at most about 2% of the graded frames (30 to 50 blasts a half, each < 1 s)
BAND_DB = 3.0       # broadband crowd noise leaves the 2 to 4 kHz band about 2 dB below its neighbours; a whistle lifts it
MIN_FRAMES = 5      # 0.2 s: a blast is longer
WINDOW_FRAMES = 12  # 0.5 s of dead after a whistle (the source's, D.3)


def _check(features, graded_mask):
    features = np.asarray(features)
    graded_mask = np.asarray(graded_mask)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must be (n_frames, 2): tonal_db and band_db per frame, got shape {}".format(
            features.shape))
    if graded_mask.shape != (len(features),):
        raise ValueError("graded_mask must have one flag per frame ({} of them), got shape {}".format(
            len(features), graded_mask.shape))
    return features, graded_mask.astype(bool)


def tonal_threshold(features, graded_mask):
    """The tonal_db a frame must reach: the TONAL_QUANTILE-th percentile of tonal_db over the graded frames, in 0.01 dB.

    Label-free and read from the features alone. R018's fixed 15 dB sat inside the crowd noise and fired 340 times;
    a whistle cannot occupy more than about 2% of the graded frames, so the match's own 98th percentile is the lowest
    level that is consistent with that duty cycle (R018b).
    """
    features, graded = _check(features, graded_mask)
    if not graded.any():
        raise ValueError("graded_mask has no graded frame: there is no tonal_db to take a percentile of")
    return round(float(np.percentile(features[graded, 0], TONAL_QUANTILE)), 2)


def runs(features, graded_mask):
    """(first, stop): int64 arrays, the first frame and the frame after the last of each blast (module docstring)."""
    features, graded = _check(features, graded_mask)
    tonal = graded & (features[:, 0] >= tonal_threshold(features, graded)) & (features[:, 1] >= BAND_DB)
    edges = np.diff(np.r_[False, tonal, False].astype(np.int8))
    first, stop = np.flatnonzero(edges == 1).astype(np.int64), np.flatnonzero(edges == -1).astype(np.int64)
    long_enough = stop - first >= MIN_FRAMES
    return first[long_enough], stop[long_enough]


def detect(features, graded_mask):
    """The int64 first frames w of the runs of at least MIN_FRAMES graded frames with both features at threshold."""
    return runs(features, graded_mask)[0]
