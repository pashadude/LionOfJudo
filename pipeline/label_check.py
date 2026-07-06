#!/usr/bin/env python3
"""
Label sanity check: does a clip's FILE NAME contradict its FOLDER?

Compilation titles are the classic poison: "Uki Goshi and Drop Ouchi Gari
from the World circuit.webm" sitting in kouchi-gari/ trains the wrong class.
The vocabulary is the dataset's own folder names, so it grows with the
corpus.

Matching detail: names are normalized to bare alphanumerics and matched as
substrings with longest-span-wins, so "kouchigari" in a title does NOT also
count as "ouchigari" (which it contains).

    python -m pipeline.label_check          # audit the whole dataset
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "dataset"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_vocab(dataset_dir: Path = DATASET_DIR) -> dict[str, str]:
    """normalized-name -> canonical folder name, longest first."""
    vocab = {}
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("youtube_"):
            vocab[_norm(d.name)] = d.name
    return dict(sorted(vocab.items(), key=lambda kv: -len(kv[0])))


def techniques_in_name(filename: str, vocab: dict[str, str]) -> set[str]:
    """Folder names mentioned in the filename (longest-span-wins)."""
    text = _norm(Path(filename).stem)
    taken: list[tuple[int, int]] = []
    found: set[str] = set()
    for key, canon in vocab.items():  # longest first
        start = 0
        while True:
            i = text.find(key, start)
            if i < 0:
                break
            span = (i, i + len(key))
            if not any(s < span[1] and span[0] < e for s, e in taken):
                taken.append(span)
                found.add(canon)
            start = i + 1
    return found


def check_clip(clip: Path, vocab: dict[str, str]) -> str | None:
    """Return a reason string if the filename contradicts the folder,
    else None (clip is fine or filename carries no technique info).

    A mentioned VARIANT of the folder's waza is not a contradiction:
    'Zantaraia Uchi-mata' belongs in uchi-mata/ even though a more
    specific zantaraia-uchi-mata folder exists."""
    folder = clip.parent.name
    mentioned = techniques_in_name(clip.name, vocab)
    if not mentioned or folder in mentioned:
        return None
    nf = _norm(folder)
    if any(nf in _norm(m) or _norm(m) in nf for m in mentioned):
        return None
    return f"filename mentions {sorted(mentioned)} but sits in '{folder}'"


def main() -> None:
    vocab = build_vocab()
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    n_checked = n_flagged = 0
    for d in sorted(DATASET_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("youtube_"):
            continue
        for clip in sorted(d.iterdir()):
            if clip.suffix.lower() not in exts:
                continue
            n_checked += 1
            reason = check_clip(clip, vocab)
            if reason:
                n_flagged += 1
                print(f"SUSPECT  {clip.relative_to(REPO_ROOT)}\n"
                      f"         {reason}")
    print(f"\n{n_checked} clips checked, {n_flagged} suspects")


if __name__ == "__main__":
    main()
