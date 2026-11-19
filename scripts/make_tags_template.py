"""Generate editable tags.csv files for YOUR OWN sfx folder.

Walks <folder> recursively; in every directory with audio files (and no
tags.csv yet) writes a prefilled tags.csv:
  filename,category,tags,description
- category = directory name
- tags     = tokens from the filename
- description = empty (fill in Excel / notepad)

The backend reads these on scan (per-directory). UTF-8 with BOM so Excel
opens Cyrillic correctly.

Usage:
  python scripts/make_tags_template.py "D:\\MySFX"
  python scripts/make_tags_template.py ./my_sfx --force   # overwrite existing
"""
from __future__ import annotations

import csv
import os
import re
import sys

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aif", ".aiff"}


def _tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-zа-яё0-9']+", name.lower())
            if len(t) > 1 and not t.isdigit()]


def main(folder: str, force: bool = False) -> None:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)
    made, skipped = 0, 0
    for root, _, names in os.walk(folder):
        audio = sorted(n for n in names
                       if os.path.splitext(n)[1].lower() in AUDIO_EXTS)
        if not audio:
            continue
        out = os.path.join(root, "tags.csv")
        if os.path.exists(out) and not force:
            print(f"skip (exists): {out}")
            skipped += 1
            continue
        cat = os.path.basename(root) or "Без категории"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["filename", "category", "tags", "description"])
            for n in audio:
                stem = os.path.splitext(n)[0]
                w.writerow([n, cat, ",".join(_tokens(stem)), ""])
        print(f"wrote {out} ({len(audio)} files)")
        made += 1
    print(f"\nDone: {made} tags.csv created, {skipped} skipped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], force="--force" in sys.argv[2:])
