"""The tactical video's audio track as two per-frame band features, cached under <data dir>/cache/audio/ (P10).

The referee's whistle is a loud, nearly pure tone in the 2-4 kHz octave. This module does the audio half of seeing
it: it decodes the mp4's audio with the pinned ffmpeg (imageio-ffmpeg, so every teammate decodes with the same
binary) and reduces it to one row of features per video frame. What to do with the features (the detector, the
`alive` evidence) lives elsewhere; keeping the decode here, cached and exactly reproducible, is what lets those
be iterated on without paying the decode again.

Alignment
---------
Row k is video frame k, i.e. the audio centred on t = (k + 0.5) / 25 s, with no offset (docs/results.md A04). At
16 kHz a frame is HOP = 640 samples, and its analysis window is WINDOW = 1024 samples centred on the frame's own 640:
samples [k*HOP - 192, k*HOP + 832), zero-padded where that falls outside the signal. Nothing is shifted to
"compensate" for a codec: the aac priming delay is the container's business, and ffmpeg already removes it.

Features
--------
Per window: a Hann-weighted power spectrum (15.625 Hz bins) and two numbers in dB, both with a 1e-12 floor inside
the log so that digital silence gives finite values:

    tonal_db   the 2-4 kHz band's strongest bin over its median bin. A whistle is one narrow peak, speech and crowd
               are broadband, so this separates them wherever the crowd's level is.
    band_db    the band's mean power over the mean power of the octaves either side (1-2 and 4-6 kHz). It asks
               whether the energy *moves into* the band, which a level-only feature cannot.

Both are rounded to 0.01 dB. That is not for tidiness: the downstream thresholds then cannot straddle a one-ulp
difference between two machines' FFTs, which is what keeps the CSV regenerable byte for byte (T20-T21).

Exactness of the chunked decode
-------------------------------
A 90-minute match is 1.6 GB of float32 at 16 kHz. It is read in CHUNK_S-second pieces, and each piece's frames are
computed only once their whole window has arrived, with the unconsumed tail carried over. Every frame is a
row-wise FFT of the very same 1024 samples as in a single pass, so the chunked result equals `features_from_samples`
on the whole signal byte for byte (checked by selftest, with the chunk patched down to one second).
"""

import hashlib
import json
import os
import platform
import subprocess
import tempfile

import numpy as np

# Params
SAMPLE_RATE = 16000                          # Hz; the whistle's band tops out far below 8 kHz
HOP = 640                                    # samples per video frame at 25 fps
WINDOW = 1024                                # samples, Hann (periodic): 15.625 Hz bins
BAND = (2000.0, 4000.0)                      # Hz, [low, high): the whistle's fundamental sits here
NEIGHBOURS = ((1000.0, 2000.0), (4000.0, 6000.0))  # Hz, the octaves either side
FEATURE_VERSION = 1                          # bump when the features or their meaning change; the key then changes
FEATURE_NAMES = ("tonal_db", "band_db")
CHUNK_S = 60                                 # seconds of audio per read
FLOOR = 1e-12                                # inside the logs: silence gives finite values
DECIMALS = 2                                 # 0.01 dB
FOLDER = "audio"
MANIFEST = "manifest.json"
ARRAY = "features.npy"
SOURCES = "source_hashes.json"               # mp4 path -> {size, mtime_ns, sha256}, so the 1.6 GB is hashed once
HASH_CHUNK = 1 << 20
BLOCK_FRAMES = 2048                          # frames per FFT batch: 2048 x 1024 float64 is 16 MB
LEAD = (WINDOW - HOP) // 2                   # samples a window reaches before its frame's own 640


def features_from_samples(samples, n_frames):
    """(n_frames, 2) float64 of [tonal_db, band_db] from a mono signal at SAMPLE_RATE, in one pass.

    Frames past the end of the signal see zeros (and so are as quiet as silence).
    """
    return features_from_stream([np.asarray(samples)], n_frames)


def features_from_stream(chunks, n_frames):
    """The same features from an iterable of consecutive sample arrays, never holding more than a chunk plus a window.

    A frame is computed as soon as the samples its window needs have arrived; the samples no later window reaches
    are dropped. Any split of the signal into chunks gives the same bytes.
    """
    n_frames = _check_frames(n_frames)
    out = np.zeros((n_frames, len(FEATURE_NAMES)), dtype=np.float64)
    buffer = np.zeros(0, dtype=np.float64)
    base = 0      # absolute sample index of buffer[0]
    done = 0      # frames computed so far
    for chunk in chunks:
        if done >= n_frames:
            continue  # the audio is longer than the frames asked for: drain it without keeping it
        buffer = np.concatenate([buffer, np.asarray(chunk, dtype=np.float64)])
        reachable = (base + len(buffer) + LEAD - WINDOW) // HOP + 1  # frames whose whole window has arrived
        done = _compute(out, buffer, base, done, min(n_frames, max(reachable, done)))
        drop = done * HOP - LEAD - base
        if drop > 0:
            buffer, base = buffer[drop:], base + drop
    if done < n_frames:  # the signal ended: what is left of every remaining window is zeros
        need = (n_frames - 1) * HOP - LEAD + WINDOW - base
        if need > len(buffer):
            buffer = np.concatenate([buffer, np.zeros(need - len(buffer))])
        _compute(out, buffer, base, done, n_frames)
    return out


def _compute(out, buffer, base, first, stop):
    """Fill out[first:stop] from `buffer`, whose element 0 is absolute sample `base`; returns stop."""
    if stop <= first:
        return first
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WINDOW) / WINDOW)
    bins = np.arange(WINDOW // 2 + 1) * (SAMPLE_RATE / WINDOW)
    band = (bins >= BAND[0]) & (bins < BAND[1])
    neighbours = np.zeros_like(band)
    for low, high in NEIGHBOURS:
        neighbours |= (bins >= low) & (bins < high)
    offsets = np.arange(WINDOW)
    padded = np.concatenate([buffer, np.zeros(1)])  # index len(buffer) is a zero, where every outside index points
    for k in range(first, stop, BLOCK_FRAMES):
        top = min(stop, k + BLOCK_FRAMES)
        index = (np.arange(k, top) * HOP - LEAD - base)[:, None] + offsets
        index = np.where((index >= 0) & (index < len(buffer)), index, len(buffer))
        power = np.abs(np.fft.rfft(padded[index] * window, axis=1)) ** 2
        in_band = power[:, band]
        tonal = 10.0 * np.log10((in_band.max(axis=1) + FLOOR) / (np.median(in_band, axis=1) + FLOOR))
        ratio = 10.0 * np.log10((in_band.mean(axis=1) + FLOOR) / (power[:, neighbours].mean(axis=1) + FLOOR))
        out[k:top, 0] = np.round(tonal, DECIMALS)
        out[k:top, 1] = np.round(ratio, DECIMALS)
    return stop


def read_in_chunks(stream):
    """Yield float32 arrays of CHUNK_S seconds each from a binary stream of little-endian float32 samples."""
    size = CHUNK_S * SAMPLE_RATE * 4
    pending = b""
    while True:
        data = stream.read(size)
        if not data:
            break
        data = pending + data
        usable = len(data) - len(data) % 4  # a read can end inside a sample; carry the odd bytes over
        pending = data[usable:]
        if usable:
            yield np.frombuffer(data[:usable], dtype="<f4")


def decode_features(mp4_path, n_frames):
    """(n_frames, 2) features of the mp4's audio, decoded with the pinned ffmpeg to mono 16 kHz float32.

    ValueError naming the file when it has no audio stream; RuntimeError with ffmpeg's own message if it fails.
    """
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError("{}: no such file".format(mp4_path))
    # `ffmpeg -i file` with no output exits non-zero, but prints the streams first: the cheapest way to ask.
    probe = subprocess.run([exe, "-nostdin", "-hide_banner", "-i", mp4_path], capture_output=True)
    if b"Audio:" not in probe.stderr:
        raise ValueError("{} has no audio stream, so there is nothing to decode (P10 needs the tactical video's "
                         "own sound)".format(mp4_path))
    command = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", mp4_path, "-vn", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-threads", "1", "-"]
    with tempfile.TemporaryFile() as errors:  # a file, not a pipe: nothing to deadlock on while stdout is read
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            features = features_from_stream(read_in_chunks(process.stdout), n_frames)
        finally:
            process.stdout.close()
            code = process.wait()
        if code != 0:
            errors.seek(0)
            raise RuntimeError("ffmpeg failed on {} (exit {}): {}".format(
                mp4_path, code, errors.read().decode("utf-8", "replace").strip()[-500:]))
    return features


def load_or_build(mp4_path, n_frames, cache_dir):
    """The features, read-only: from <cache_dir>/audio/<key>/ when it is fresh, otherwise decoded and stored.

    The key is the sha256 of the mp4's own sha256 and every parameter the features depend on, as canonical JSON, so
    a replaced video or a changed parameter is a different folder, and the old one is left in place. The mp4's hash
    is itself remembered by (size, mtime) in <cache_dir>/audio/source_hashes.json: the file is 1.6 GB, and hashing it
    on every run would cost more than reading the features. The manifest holds the features' own sha256, checked on
    every load, which is also the determinism check (two decodes must give the same hash).
    """
    n_frames = _check_frames(n_frames)
    root = os.path.join(cache_dir, FOLDER)
    video = _video_hash(mp4_path, root)
    key = hashlib.sha256(json.dumps({
        "video_sha256": video, "feature_version": FEATURE_VERSION, "sample_rate": SAMPLE_RATE, "hop": HOP,
        "window": WINDOW, "band": list(BAND), "neighbours": [list(pair) for pair in NEIGHBOURS],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    folder = os.path.join(root, key)
    features = _load(folder, n_frames)
    if features is None:
        features = decode_features(mp4_path, n_frames)
        _save(folder, features, {
            "video_sha256": video, "feature_version": FEATURE_VERSION, "n_frames": n_frames,
            "names": list(FEATURE_NAMES), "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "python": platform.python_version(), "numpy": np.__version__, "ffmpeg": _ffmpeg_version(),
        })
    features.flags.writeable = False
    return features


def _check_frames(n_frames):
    if not isinstance(n_frames, (int, np.integer)) or isinstance(n_frames, bool) or n_frames < 1:
        raise ValueError("n_frames must be a positive integer, got {!r}".format(n_frames))
    return int(n_frames)


def _video_hash(path, root):
    """sha256 of the file, from root/source_hashes.json when its size and mtime are unchanged."""
    stat = os.stat(path)
    table_path = os.path.join(root, SOURCES)
    table = {}
    if os.path.isfile(table_path):
        try:
            with open(table_path, encoding="utf-8") as handle:
                table = json.load(handle)
        except (OSError, ValueError):
            table = {}
    name = os.path.abspath(path)
    known = table.get(name)
    if known and known.get("size") == stat.st_size and known.get("mtime_ns") == stat.st_mtime_ns:
        return known["sha256"]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    table[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}
    os.makedirs(root, exist_ok=True)
    with open(table_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(table, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return table[name]["sha256"]


def _load(folder, n_frames):
    """The cached features, or None when absent, incomplete, of another length, or not the hash its manifest names."""
    manifest_path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        features = np.load(os.path.join(folder, ARRAY), allow_pickle=False)  # in memory: Windows locks a mapped file
    except (OSError, ValueError, EOFError):
        return None
    if features.shape != (n_frames, len(FEATURE_NAMES)) or features.dtype != np.float64:
        return None
    if hashlib.sha256(features.tobytes()).hexdigest() != manifest.get("features_sha256"):
        return None
    return features


def _save(folder, features, manifest):
    """Array first, manifest last, the old manifest removed up front: the manifest marks a cache as complete."""
    os.makedirs(folder, exist_ok=True)
    manifest_path = os.path.join(folder, MANIFEST)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    np.save(os.path.join(folder, ARRAY), features, allow_pickle=False)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _ffmpeg_version():
    import imageio_ffmpeg
    done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], capture_output=True)
    first = done.stdout.decode("utf-8", "replace").splitlines()
    return first[0] if first else "unknown"
