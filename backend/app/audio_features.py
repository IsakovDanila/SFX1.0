"""Fast spectral analysis with Librosa.

Kept deliberately cheap: no yin/hpss/beat-tracking (each costs seconds on
long files, and their outputs were never shown in the UI).
Typical cost: ~0.1-0.3 s per SFX file on CPU.

Only the first `max_seconds` (default 60 s) are decoded for spectral stats;
`duration` always reflects the full file (via soundfile header, no decode).
"""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

ANALYSIS_SR = 22050


def analyze(path: str, max_seconds: float = 60.0) -> dict:
    try:
        full_duration = float(sf.info(path).duration)
    except Exception:
        full_duration = 0.0
    y, sr = librosa.load(path, sr=ANALYSIS_SR, mono=True, duration=max_seconds)
    duration = full_duration or (float(len(y) / sr) if len(y) else 0.0)
    if len(y) < 16:
        return {"duration": round(duration, 3), "sr": int(sr), "error": "too_short"}

    S = np.abs(librosa.stft(y))
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    flatness = librosa.feature.spectral_flatness(S=S)

    frame_e = librosa.feature.rms(y=y)[0]
    if len(frame_e) >= 8:
        early = float(np.mean(frame_e[: max(1, len(frame_e) // 8)]) + 1e-12)
        tail = float(np.mean(frame_e[-max(1, len(frame_e) // 4):]))
        reverb_proxy = float(np.clip(tail / early, 0, 1))
    else:
        reverb_proxy = 0.0

    return {
        "duration": round(duration, 3),
        "sr": int(sr),
        "rms_energy": round(rms, 5),
        "peak": round(peak, 5),
        "crest_db": round(float(20 * np.log10(peak / (rms + 1e-12))), 2),
        "spectral_centroid_mean": round(float(np.mean(centroid)), 1),
        "spectral_rolloff_mean": round(float(np.mean(rolloff)), 1),
        "zcr_mean": round(float(np.mean(zcr)), 5),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 4),
        "mfcc_mean": [round(float(v), 4) for v in np.mean(mfcc, axis=1)],
        "reverb_proxy": round(reverb_proxy, 4),
    }
