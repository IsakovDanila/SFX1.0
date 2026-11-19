"""CLAP zero-shot auto-tagging: sound -> words, no training needed.

During a CLAP-mode scan each file's audio embedding is compared against
text embeddings of a label vocabulary ("This is a sound of ..."). Top
matches become `auto_tags` (with scores) and are merged into `tags` so
keyword search finds them immediately.

Honest limits:
- speech is labeled by HOW it sounds (male/female/shout/laugh), not WHAT
  is said -- transcribing meme phrases needs ASR (Whisper), not CLAP;
- music genres are broad guesses; SFX labels are the most reliable.

Tuning: SFX_AUTOTAG_TOPK (default 3), SFX_AUTOTAG_THRESHOLD (default 0.12),
SFX_LABELS_PATH (optional JSON file with [{"en","ru","group"}] to replace
the built-in vocabulary).
"""
from __future__ import annotations

import json
import os

import numpy as np

PROMPT = "This is a sound of {}."

# (english label for CLAP, russian tag, group)
LABELS: list[tuple[str, str, str]] = [
    # ---- speech / meme voices: HOW it sounds ----
    ("male speech, man talking", "мужская речь", "speech"),
    ("female speech, woman talking", "женская речь", "speech"),
    ("child voice, kid talking", "детский голос", "speech"),
    ("crowd talking, people chattering", "гул толпы", "speech"),
    ("shouting, screaming", "крик", "speech"),
    ("whispering", "шёпот", "speech"),
    ("laughter, laughing", "смех", "speech"),
    ("crying, sobbing", "плач", "speech"),
    ("singing", "пение", "speech"),
    ("choir singing", "хор", "speech"),
    ("cartoon character voice", "мультяшный голос", "speech"),
    ("robotic voice, text to speech", "роботизированный голос", "speech"),
    ("telephone conversation, phone call", "телефонный разговор", "speech"),
    ("speech with background music", "речь с музыкой", "speech"),
    ("coughing", "кашель", "speech"),
    ("sneezing", "чихание", "speech"),
    ("burping", "отрыжка", "speech"),
    # ---- music (broad) ----
    ("music", "музыка", "music"),
    ("electronic dance music", "электроника", "music"),
    ("hip hop music", "хип-хоп", "music"),
    ("rock music with electric guitar", "рок", "music"),
    ("pop music", "поп-музыка", "music"),
    ("classical orchestral music", "классика", "music"),
    ("jazz music", "джаз", "music"),
    ("8-bit chiptune video game music", "8-битная музыка", "music"),
    ("dramatic epic cinematic music", "эпичная музыка", "music"),
    ("funny comedic music", "смешная музыка", "music"),
    ("sad melancholic music", "грустная музыка", "music"),
    ("tense suspense music", "напряжённая музыка", "music"),
    ("heavy bass drop", "бас-дроп", "music"),
    ("piano melody", "пианино", "music"),
    ("acoustic guitar", "гитара", "music"),
    ("drum beat", "барабаны", "music"),
    # ---- SFX ----
    ("hands clapping, applause", "хлопки", "sfx"),
    ("finger snapping", "щелчок пальцами", "sfx"),
    ("explosion, bomb blast", "взрыв", "sfx"),
    ("gunshot", "выстрел", "sfx"),
    ("punch, thud hit", "удар", "sfx"),
    ("glass breaking", "звон стекла", "sfx"),
    ("footsteps walking", "шаги", "sfx"),
    ("door slamming", "хлопок двери", "sfx"),
    ("ui click, mouse click", "клик", "sfx"),
    ("phone notification ding", "уведомление", "sfx"),
    ("air horn", "горн", "sfx"),
    ("sad trombone wah wah", "грустный тромбон", "sfx"),
    ("record scratch", "скретч пластинки", "sfx"),
    ("drum rimshot, ba dum tss", "римшот", "sfx"),
    ("buzzer, wrong answer buzz", "зуммер ошибки", "sfx"),
    ("success fanfare, victory jingle", "фанфары победы", "sfx"),
    ("cash register cha-ching", "касса", "sfx"),
    ("camera shutter click", "затвор камеры", "sfx"),
    ("siren, emergency alarm", "сирена", "sfx"),
    ("car horn honking", "автомобильный гудок", "sfx"),
    ("laser zap, sci-fi shot", "лазер", "sfx"),
    ("cartoon boing sound effect", "мультяшный эффект", "sfx"),
    ("slide whistle", "свистулька-слайд", "sfx"),
    ("thunder", "гром", "sfx"),
    ("rain falling", "дождь", "sfx"),
    ("wind blowing", "ветер", "sfx"),
    ("fire crackling", "треск огня", "sfx"),
    ("water splash", "всплеск воды", "sfx"),
    ("bubbles, bubbling water", "бульканье", "sfx"),
    ("heartbeat", "сердцебиение", "sfx"),
    ("clock ticking", "тиканье часов", "sfx"),
    ("radio static noise", "радиопомехи", "sfx"),
    ("crowd cheering", "ликование толпы", "sfx"),
    ("crowd booing", "свист толпы", "sfx"),
]


def load_labels() -> list[tuple[str, str, str]]:
    path = os.getenv("SFX_LABELS_PATH", "")
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out = [(d["en"], d.get("ru", d["en"]), d.get("group", ""))
                   for d in data if d.get("en")]
            if out:
                print(f"[autotag] custom vocabulary: {len(out)} labels from {path}")
                return out
        except (OSError, ValueError, KeyError) as e:
            print(f"[autotag] bad SFX_LABELS_PATH ({e}), using built-in vocab")
    return LABELS


class ZeroShotTagger:
    """Rank vocabulary labels against an audio vector by cosine sim."""

    def __init__(self, embedder) -> None:
        from . import embeddings as E  # local import: no cycles
        self._cosine = E.cosine
        self.labels = load_labels()
        self.top_k = int(os.getenv("SFX_AUTOTAG_TOPK", "3"))
        self.threshold = float(os.getenv("SFX_AUTOTAG_THRESHOLD", "0.12"))
        print(f"[autotag] encoding {len(self.labels)} labels "
              f"(top_k={self.top_k}, threshold={self.threshold})...")
        self._text_vecs = [embedder.text_vector(PROMPT.format(en))
                            for en, _, _ in self.labels]
        print("[autotag] label embeddings ready")

    def tag(self, audio_vec: np.ndarray) -> list[dict]:
        scored = sorted(
            ((self._cosine(audio_vec, tv), i)
             for i, tv in enumerate(self._text_vecs)),
            key=lambda x: -x[0])
        out = []
        for sim, i in scored[:self.top_k]:
            if sim < self.threshold:
                break
            en, ru, group = self.labels[i]
            out.append({"en": en, "ru": ru, "group": group,
                        "score": round(float(sim), 4)})
        return out
