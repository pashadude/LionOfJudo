#!/usr/bin/env python3
"""
Reference bank: nearest-neighbor technique matching against the FULL
dataset/ catalog (all ~190 waza, even classes with a single demo clip).

Complements the trained classifier: the classifier is precise on the few
techniques with many samples; the bank gives breadth — every detected throw
is compared against every reference clip and the top matches are reported.

Each waza carries a category: throw (nage-waza) or hold (katame-waza /
ne-waza), derived from its name.

    python -m pipeline.reference_bank --build     # after feature extraction
    python -m pipeline.reference_bank --query dataset/o-goshi/clip.npz
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "dataset"
BANK_PATH = REPO_ROOT / "models" / "reference_bank.npz"

# waza -> throw|hold: katame-waza names are recognisable from components
HOLD_PATTERN = re.compile(
    r"gatame|jime|shime|garami|hishigi|osaekomi|escape|sankaku|katame")


def waza_category(name: str) -> str:
    """'throw' (nage-waza) or 'hold' (katame-waza / ne-waza) from the slug."""
    return "hold" if HOLD_PATTERN.search(name.lower()) else "throw"


def build_bank(dataset_dir: Path = DATASET_DIR) -> dict:
    feats, labels = [], []
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("youtube_"):
            continue
        for npz in sorted(d.glob("*.npz")):
            try:
                feats.append(np.load(npz)["stats"])
                labels.append(d.name)
            except Exception:
                pass
    if not feats:
        raise SystemExit("no .npz features found — run "
                         "'python -m pipeline.pose_features dataset/' first")

    X = np.nan_to_num(np.stack(feats), nan=0.0, posinf=0.0, neginf=0.0)
    # L2-normalize rows -> cosine similarity is a dot product
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    uniq = sorted(set(labels))
    cats = {u: waza_category(u) for u in uniq}
    BANK_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        BANK_PATH, X=X.astype(np.float32),
        labels=np.array(labels),
        categories=json.dumps(cats))

    n_throw = sum(1 for u in uniq if cats[u] == "throw")
    n_hold = len(uniq) - n_throw
    print(f"bank: {len(X)} reference clips, {len(uniq)} waza "
          f"({n_throw} throw / {n_hold} hold) -> {BANK_PATH.name}")
    return {"clips": len(X), "waza": len(uniq)}


def load_bank():
    if not BANK_PATH.exists():
        return None
    d = np.load(BANK_PATH, allow_pickle=False)
    return {"X": d["X"], "labels": d["labels"],
            "categories": json.loads(str(d["categories"]))}


def match(bank, stats_vector: np.ndarray, top_k: int = 3) -> list[dict]:
    """Top-k waza by max cosine similarity over each waza's reference clips."""
    q = np.nan_to_num(stats_vector.astype(np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = bank["X"] @ q

    best_per_waza: dict[str, float] = {}
    for lab, s in zip(bank["labels"], sims):
        lab = str(lab)
        if s > best_per_waza.get(lab, -1):
            best_per_waza[lab] = float(s)

    top = sorted(best_per_waza.items(), key=lambda kv: -kv[1])[:top_k]
    return [{"waza": w, "similarity": round(s, 3),
             "category": bank["categories"].get(w, "throw")}
            for w, s in top]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--query", type=Path,
                   help=".npz feature file to match against the bank")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    if args.build:
        build_bank()
    if args.query:
        bank = load_bank()
        if bank is None:
            raise SystemExit("no bank — run with --build first")
        stats = np.load(args.query)["stats"]
        for m in match(bank, stats, args.top_k):
            print(f"  {m['similarity']:.3f}  {m['waza']}  [{m['category']}]")


if __name__ == "__main__":
    main()
