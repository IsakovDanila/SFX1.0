"""Generate a synthetic demo SFX dataset (no external files needed).

40 wav files, 8 categories x 5 variants, 22050 Hz mono:
  footstep_wood / explosion_cinematic / interface_click / metal_hit /
  glass_break / whoosh / thunder / gunshot
+ tags.csv with reference tags (deck task 1.1: CSV with ground-truth tags).

Run:  python scripts/generate_demo_sounds.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import soundfile as sf
from scipy import signal as sig

SR = 22050
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "demo_sfx")

TAGS = {
    "footstep_wood": "footstep,wood,wooden floor,steps,walking,stomp",
    "explosion_cinematic": "explosion,cinematic,boom,blast,epic",
    "interface_click": "click,ui,interface,button,blip",
    "metal_hit": "metal,metallic,clang,impact,steel",
    "glass_break": "glass,break,shatter,smash,crash",
    "whoosh": "whoosh,sweep,transition,flyby,wind",
    "thunder": "thunder,storm,rumble,lightning",
    "gunshot": "gunshot,gun,firearm,shot,pistol",
}


def _norm(y: np.ndarray, peak: float = 0.85) -> np.ndarray:
    m = float(np.max(np.abs(y))) or 1.0
    return (y / m * peak).astype(np.float32)


def _env(n: int, attack: float = 0.005, decay: float = 3.0) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t * decay)


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    b = np.cumsum(w)
    b -= np.linspace(b[0], b[-1], n)
    return b / (np.max(np.abs(b)) + 1e-12)


def _lowpass(y: np.ndarray, cutoff: float) -> np.ndarray:
    sos = sig.butter(4, cutoff / (SR / 2), btype="low", output="sos")
    return sig.sosfilt(sos, y).astype(np.float32)


def _highpass(y: np.ndarray, cutoff: float) -> np.ndarray:
    sos = sig.butter(4, cutoff / (SR / 2), btype="high", output="sos")
    return sig.sosfilt(sos, y).astype(np.float32)


def footstep_wood(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(0.28, 0.4))
    t = np.arange(n) / SR
    f = rng.uniform(65, 95)
    thump = np.sin(2 * np.pi * f * t) * np.exp(-t * rng.uniform(18, 26))
    body = 0.4 * np.sin(2 * np.pi * f * 2.4 * t) * np.exp(-t * 30)
    tap = _lowpass(rng.standard_normal(n), 900) * np.exp(-t * 90) * 0.8
    return _norm(thump + body + tap)


def explosion_cinematic(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(1.8, 2.6))
    t = np.arange(n) / SR
    boom = _lowpass(_brown(n, rng), rng.uniform(220, 380))
    boom *= _env(n, attack=0.01, decay=rng.uniform(1.4, 2.0))
    sub = np.sin(2 * np.pi * rng.uniform(38, 50) * t) * np.exp(-t * 2.2) * 0.7
    crack = _lowpass(rng.standard_normal(n), 3000) * np.exp(-t * 60) * 0.9
    return _norm(boom + sub + crack)


def interface_click(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    click = rng.standard_normal(n) * np.exp(-t * 400) * 0.9
    f = rng.uniform(1700, 2400)
    blip = np.sin(2 * np.pi * f * t) * np.exp(-t * rng.uniform(40, 60)) * 0.6
    return _norm(click + blip, 0.7)


def metal_hit(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(1.0, 1.4))
    t = np.arange(n) / SR
    f0 = rng.uniform(200, 380)
    y = np.zeros(n, dtype=np.float32)
    for ratio, amp, dec in [(1.0, 1.0, 4.5), (2.76, 0.6, 6.0),
                            (5.40, 0.4, 8.0), (8.93, 0.25, 10.0)]:
        y += amp * np.sin(2 * np.pi * f0 * ratio * t) * np.exp(-t * dec)
    y += _highpass(rng.standard_normal(n), 4000) * np.exp(-t * 80) * 0.25
    return _norm(y)


def glass_break(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(0.7, 0.9))
    y = np.zeros(n, dtype=np.float32)
    for _ in range(int(rng.integers(4, 7))):
        start = int(rng.uniform(0, 0.35) * SR)
        m = int(SR * rng.uniform(0.08, 0.2))
        if start + m >= n:
            continue
        t = np.arange(m) / SR
        burst = _highpass(rng.standard_normal(m), 2500) * np.exp(-t * 35)
        f = rng.uniform(2800, 6200)
        ping = np.sin(2 * np.pi * f * t) * np.exp(-t * 30) * 0.7
        y[start:start + m] += (burst + ping) * rng.uniform(0.5, 1.0)
    return _norm(y)


def whoosh(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(0.8, 1.1))
    noise = rng.standard_normal(n).astype(np.float32)
    lo = _lowpass(noise, 800)
    hi = _highpass(noise, 2500)
    t = np.linspace(0, 1, n)
    k = np.sin(np.pi * np.clip(t, 0, 1)) ** 1.5  # rise then fall
    y = (1 - k) * lo + k * hi
    return _norm(y * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.7 + 1e-6)


def thunder(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(2.2, 3.0))
    t = np.arange(n) / SR
    rumble = _lowpass(_brown(n, rng), rng.uniform(120, 200))
    attack = np.clip(t / rng.uniform(0.15, 0.4), 0, 1)
    y = rumble * attack * np.exp(-t * rng.uniform(0.9, 1.3))
    crack_at = int(rng.uniform(0.05, 0.3) * SR)
    m = min(int(0.25 * SR), n - crack_at)
    tc = np.arange(m) / SR
    y[crack_at:crack_at + m] += _lowpass(
        rng.standard_normal(m), 2500) * np.exp(-tc * 25) * 0.8
    return _norm(y)


def gunshot(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * rng.uniform(0.5, 0.7))
    t = np.arange(n) / SR
    snap = rng.standard_normal(n) * np.exp(-t * rng.uniform(45, 60))
    body = _lowpass(rng.standard_normal(n), 1200) * np.exp(-t * 14) * 0.8
    thump = np.sin(2 * np.pi * rng.uniform(55, 70) * t) * np.exp(-t * 22) * 0.7
    return _norm(snap + body + thump)


GENS = {"footstep_wood": footstep_wood, "explosion_cinematic": explosion_cinematic,
        "interface_click": interface_click, "metal_hit": metal_hit,
        "glass_break": glass_break, "whoosh": whoosh, "thunder": thunder,
        "gunshot": gunshot}


def main(variants: int = 5) -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cat, fn in GENS.items():
        for i in range(variants):
            rng = np.random.default_rng(hash((cat, i)) % (2 ** 31))
            y = fn(rng)
            name = f"{cat}_{i:02d}.wav"
            sf.write(os.path.join(OUT, name), y, SR, subtype="PCM_16")
            rows.append({"filename": name, "category": cat, "tags": TAGS[cat]})
            print(f"wrote {name} ({len(y) / SR:.2f}s)")
    with open(os.path.join(OUT, "tags.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "category", "tags"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone: {len(rows)} files -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
