"""Validation croisee 5-fold stratifiee par patient."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import KFold


def make_5fold_splits(items: Sequence[dict],
                      n_folds: int = 5,
                      seed: int = 42,
                      out_path: str | Path | None = None) -> list[dict]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(items)):
        folds.append({
            "fold": fold,
            "train": [items[i]["patient_id"] for i in train_idx],
            "val": [items[i]["patient_id"] for i in val_idx],
        })
    if out_path is not None:
        Path(out_path).write_text(json.dumps(folds, indent=2, ensure_ascii=False))
    return folds


def load_fold(folds_or_path, fold: int, items: Sequence[dict]):
    if isinstance(folds_or_path, (str, Path)):
        folds = json.loads(Path(folds_or_path).read_text())
    else:
        folds = folds_or_path
    fold_dict = folds[fold]
    train_ids = set(fold_dict["train"])
    val_ids = set(fold_dict["val"])
    train_items = [it for it in items if it["patient_id"] in train_ids]
    val_items = [it for it in items if it["patient_id"] in val_ids]
    return train_items, val_items
