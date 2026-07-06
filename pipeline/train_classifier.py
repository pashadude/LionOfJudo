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


def load_dataset(dataset_dir: Path, min_per_class: int,
                 exclude: tuple[str, ...] = ("youtube_",)
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, labels = [], [], []
    skipped_classes = []
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or any(d.name.startswith(x) for x in exclude):
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


def candidate_models():
    """Candidates tried via CV; best one is kept. Important detail: tree
    defaults like HGB's min_samples_leaf=20 silently collapse to a
    majority-class predictor on small corpora — hence the explicit values."""
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.1,
                               class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=1,
            class_weight="balanced", random_state=7),
        "hist_gb": HistGradientBoostingClassifier(
            min_samples_leaf=2, max_iter=200, max_depth=4,
            l2_regularization=1.0, random_state=7),
    }


def main() -> None:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                         cross_val_score)
    import joblib

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DATASET_DIR)
    p.add_argument("--min-per-class", type=int, default=3,
                   help="classes with fewer extracted samples are skipped")
    p.add_argument("--extract", action="store_true",
                   help="run feature extraction on the dataset first")
    p.add_argument("--eval-dir", type=Path, default=None,
                   help="held-out eval set (eval_set/<technique>/ layout, "
                        "never trained on) to score the final model against")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    if args.extract:
        from ultralytics import YOLO
        from pipeline.pose_features import extract_clip_features
        model = YOLO(str(REPO_ROOT / "yolo11x-pose.pt"))
        for clip in sorted(args.dataset.rglob("*")):
            if clip.suffix.lower() in VIDEO_EXTS and not any(
                    part.startswith("youtube_") for part in clip.parts):
                extract_clip_features(clip, model, args.device)

    X, y, labels = load_dataset(args.dataset, args.min_per_class)
    n_classes = len(labels)
    counts = np.bincount(y, minlength=n_classes)
    print(f"\n{len(X)} samples, {n_classes} classes:")
    for i, lab in enumerate(labels):
        print(f"  {lab:24s} {counts[i]:4d}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- model selection + cross-validation report -------------------------
    candidates = candidate_models()
    n_splits = int(min(5, counts.min()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
        print(f"\nmodel selection ({n_splits}-fold CV):")
        scores = {}
        for name, m in candidates.items():
            scores[name] = cross_val_score(m, X, y, cv=cv).mean()
            print(f"  {name:16s} {scores[name]:.1%}")
        best_name = max(scores, key=scores.get)
        clf = candidates[best_name]
        print(f"selected: {best_name}")
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
        clf = candidates["logreg"]
        print("\nToo few samples per class for cross-validation — "
              "training logreg anyway, but treat the model as a placeholder.")

    # ---- final fit on everything ------------------------------------------
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))
    print(f"\nsaved {MODEL_PATH.relative_to(REPO_ROOT)} "
          f"and {LABELS_PATH.relative_to(REPO_ROOT)}")

    # ---- held-out eval set (the number to trust) ---------------------------
    if args.eval_dir and args.eval_dir.exists():
        print(f"\n=== held-out eval: {args.eval_dir} ===")
        Xe, ye_raw, eval_labels = load_dataset(args.eval_dir, 1)
        known = [i for i, lab in enumerate(eval_labels) if lab in labels]
        unknown = [lab for lab in eval_labels if lab not in labels]
        if unknown:
            print(f"  (skipping techniques the model wasn't trained on: "
                  f"{', '.join(unknown)})")
        mask = np.isin(ye_raw, known)
        if not mask.any():
            print("  no eval clips match trained classes")
            return
        Xe = np.nan_to_num(Xe[mask], nan=0.0, posinf=0.0, neginf=0.0)
        ye = np.array([labels.index(eval_labels[i]) for i in ye_raw[mask]])
        pred = clf.predict(Xe)
        acc = float((pred == ye).mean())
        print(f"  accuracy on {len(ye)} real-footage clips: {acc:.1%}")
        for i, p_ in zip(ye, pred):
            mark = "✓" if i == p_ else "✗"
            print(f"    {mark} true={labels[i]:24s} pred={labels[p_]}")


if __name__ == "__main__":
    main()
