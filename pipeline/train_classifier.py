#!/usr/bin/env python3
"""
Train the technique classifier from the dataset/ corpus.

    python -m pipeline.train_classifier                 # train + CV report
    python -m pipeline.train_classifier --min-per-class 5

Reads cached .npz features (run pipeline/pose_features.py on dataset/ first,
or pass --extract to do it here). Trains HistGradientBoosting on the compact
stats features; prints stratified k-fold cross-validation per-class results
and a confusion matrix, then fits on all data and saves:

    models/technique_clf.joblib
    models/technique_labels.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "dataset"
MODEL_PATH = REPO_ROOT / "models" / "technique_clf.joblib"
LABELS_PATH = REPO_ROOT / "models" / "technique_labels.json"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_dataset(dataset_dir: Path, min_per_class: int
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, labels = [], [], []
    skipped_classes = []
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir():
            continue
        feats = []
        for clip in sorted(d.iterdir()):
            if clip.suffix.lower() not in VIDEO_EXTS:
                continue
            cache = clip.with_suffix(".npz")
            if cache.exists():
                feats.append(np.load(cache)["stats"])
        if len(feats) < min_per_class:
            skipped_classes.append((d.name, len(feats)))
            continue
        idx = len(labels)
        labels.append(d.name)
        X.extend(feats)
        y.extend([idx] * len(feats))

    for name, n in skipped_classes:
        print(f"  skipping class '{name}': only {n} extracted samples "
              f"(< {min_per_class})")
    if not X:
        raise SystemExit(
            "No usable classes. Add clips to dataset/<technique>/ and run:\n"
            "  python -m pipeline.pose_features dataset/")
    return np.stack(X), np.array(y), labels


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import joblib

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DATASET_DIR)
    p.add_argument("--min-per-class", type=int, default=3,
                   help="classes with fewer extracted samples are skipped")
    p.add_argument("--extract", action="store_true",
                   help="run feature extraction on the dataset first")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    if args.extract:
        from ultralytics import YOLO
        from pipeline.pose_features import extract_clip_features
        model = YOLO(str(REPO_ROOT / "yolo11x-pose.pt"))
        for clip in sorted(args.dataset.rglob("*")):
            if clip.suffix.lower() in VIDEO_EXTS:
                extract_clip_features(clip, model, args.device)

    X, y, labels = load_dataset(args.dataset, args.min_per_class)
    n_classes = len(labels)
    counts = np.bincount(y, minlength=n_classes)
    print(f"\n{len(X)} samples, {n_classes} classes:")
    for i, lab in enumerate(labels):
        print(f"  {lab:24s} {counts[i]:4d}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=4,
        l2_regularization=1.0, random_state=7)

    # ---- cross-validation report -----------------------------------------
    n_splits = int(min(5, counts.min()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        acc = float((y_pred == y).mean())
        print(f"\n=== {n_splits}-fold CV accuracy: {acc:.1%} ===")
        print(classification_report(y, y_pred, target_names=labels,
                                    zero_division=0))
        print("confusion matrix (rows=true, cols=predicted):")
        cm = confusion_matrix(y, y_pred)
        width = max(len(l) for l in labels)
        for i, row in enumerate(cm):
            print(f"  {labels[i]:{width}s} {row}")
        weakest = min(range(n_classes),
                      key=lambda i: (y_pred[y == i] == i).mean() if counts[i] else 1)
        print(f"\nweakest class: '{labels[weakest]}' — film more reps of it.")
    else:
        print("\nToo few samples per class for cross-validation — "
              "training anyway, but treat the model as a placeholder.")

    # ---- final fit on everything ------------------------------------------
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))
    print(f"\nsaved {MODEL_PATH.relative_to(REPO_ROOT)} "
          f"and {LABELS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
