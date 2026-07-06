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
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "dataset"
MODEL_PATH = REPO_ROOT / "models" / "technique_clf.joblib"
LABELS_PATH = REPO_ROOT / "models" / "technique_labels.json"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def group_key(clip: Path) -> str:
    """Same source video -> same CV group (rep splits, youtube ids)."""
    m = re.search(r"\[([A-Za-z0-9_-]{6,})\]", clip.stem)
    if m:
        return m.group(1)                      # youtube id
    return re.sub(r"_rep\d+$", "", clip.stem)  # rep splits share the source


def load_dataset(dataset_dir: Path, min_per_class: int,
                 exclude: tuple[str, ...] = ("youtube_",), *,
                 with_seq: bool = False):
    """Returns (X_stats, y, labels) or, with_seq, (X_stats, y, labels,
    seqs, groups)."""
    from pipeline.label_check import build_vocab, check_clip

    vocab = build_vocab(dataset_dir) if dataset_dir.exists() else {}
    X, y, labels, seqs, groups = [], [], [], [], []
    skipped_classes = []
    n_suspect = 0
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or any(d.name.startswith(x) for x in exclude):
            continue
        feats = []
        for clip in sorted(d.iterdir()):
            if clip.suffix.lower() not in VIDEO_EXTS:
                continue
            reason = check_clip(clip, vocab)
            if reason:
                n_suspect += 1
                print(f"  EXCLUDED (label suspect): {clip.name} — {reason}")
                continue
            cache = clip.with_suffix(".npz")
            if cache.exists():
                data = np.load(cache)
                feats.append((data["stats"], data["seq"], group_key(clip)))
        if len(feats) < min_per_class:
            skipped_classes.append((d.name, len(feats)))
            continue
        idx = len(labels)
        labels.append(d.name)
        for st, sq, g in feats:
            X.append(st)
            seqs.append(sq)
            groups.append(g)
            y.append(idx)

    if n_suspect:
        print(f"  ({n_suspect} clips excluded by filename/folder cross-check "
              f"— see pipeline/label_check.py)")
    for name, n in skipped_classes:
        print(f"  skipping class '{name}': only {n} extracted samples "
              f"(< {min_per_class})")
    if not X:
        raise SystemExit(
            "No usable classes. Add clips to dataset/<technique>/ and run:\n"
            "  python -m pipeline.pose_features dataset/")
    if with_seq:
        return (np.stack(X), np.array(y), labels,
                np.stack(seqs), np.array(groups))
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


def augment_flip(X_stats, y, seqs):
    """Append left/right-mirrored copies. TRAINING folds only — augmenting
    before the CV split would leak mirror-twins across folds."""
    from pipeline.pose_features import flip_seq, stats_from_seq
    X_aug = np.stack([stats_from_seq(flip_seq(s)) for s in seqs])
    return (np.concatenate([X_stats, X_aug]),
            np.concatenate([y, y]))


def grouped_cv_predict(make_clf, X, y, seqs, groups, n_splits):
    """Grouped, stratified CV with in-fold flip augmentation.
    Returns out-of-fold predictions for every sample."""
    from sklearn.model_selection import StratifiedGroupKFold

    y_pred = np.full_like(y, -1)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    for tr, te in cv.split(X, y, groups):
        Xtr, ytr = augment_flip(X[tr], y[tr], seqs[tr])
        clf = make_clf()
        clf.fit(Xtr, ytr)
        y_pred[te] = clf.predict(X[te])
    return y_pred


def main() -> None:
    from sklearn.metrics import classification_report, confusion_matrix
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

    X, y, labels, seqs, groups = load_dataset(
        args.dataset, args.min_per_class, with_seq=True)
    n_classes = len(labels)
    counts = np.bincount(y, minlength=n_classes)
    print(f"\n{len(X)} samples, {n_classes} classes, "
          f"{len(set(groups))} source-video groups:")
    for i, lab in enumerate(labels):
        print(f"  {lab:24s} {counts[i]:4d}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    seqs = np.nan_to_num(seqs, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- model selection + grouped CV with flip augmentation ---------------
    candidates = candidate_models()
    n_splits = int(min(5, counts.min()))
    if n_splits >= 2:
        print(f"\nmodel selection ({n_splits}-fold grouped CV, "
              f"flip-augmented):")
        scores, preds = {}, {}
        for name in candidates:
            p = grouped_cv_predict(lambda n=name: candidate_models()[n],
                                   X, y, seqs, groups, n_splits)
            scores[name] = float((p == y).mean())
            preds[name] = p
            print(f"  {name:16s} {scores[name]:.1%}")
        best_name = max(scores, key=scores.get)
        clf = candidates[best_name]
        print(f"selected: {best_name}")
        y_pred = preds[best_name]
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

    # ---- final fit on everything (flip-augmented) ---------------------------
    Xf, yf = augment_flip(X, y, seqs)
    clf.fit(Xf, yf)
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
