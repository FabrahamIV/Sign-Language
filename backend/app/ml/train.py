"""Training, validation, and TorchScript export script for the SignStream StaticSignMLP (PDD Milestone 2).

Follows Essential ML Best Practices:
- Checks for and handles missing/NULL values and malformed rows before modeling.
- Enforces Strict Featurization Ordering: splits raw 21x3 coordinates into train/val/test sets
  BEFORE applying translation and L2 scale normalization independently on each split.
- Evaluates macro F1-score, accuracy, and per-class confusion matrix diagnostics.
- Exports the best checkpoint (.pt and TorchScript) to `app/ml/weights/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.database import SignStreamDatabase
from app.ml.model import (
    METADATA_PATH,
    STATIC_CLASSES,
    STATIC_TORCHSCRIPT_PATH,
    STATIC_WEIGHTS_PATH,
    WEIGHTS_DIR,
    StaticSignMLP,
    build_canonical_hand_prototypes,
)
from app.vision.normalizer import normalize_hand_array


def generate_synthetic_raw_dataset(
    samples_per_class: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate raw (unnormalized) 21x3 hand landmark samples with realistic camera translation, scale, and noise.

    Because raw coordinates include arbitrary wrist translations and hand scales, normalizing AFTER
    splitting into train/val/test verifies translation and scale invariance without data leakage.
    """
    rng = np.random.default_rng(seed)
    prototypes = build_canonical_hand_prototypes()
    rows: List[Dict[str, Any]] = []

    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    for label in STATIC_CLASSES:
        base_pose = prototypes[label]  # (21, 3)
        for _ in range(samples_per_class):
            # Apply random scale factor (hand distance from camera) and random wrist translation
            scale = float(rng.uniform(0.65, 1.45))
            shift = rng.uniform(-0.18, 0.18, size=(1, 3)).astype(np.float32)
            jitter = rng.normal(0.0, 0.0085, size=(21, 3)).astype(np.float32)

            wrist = base_pose[0:1, :]
            rel = (base_pose - wrist) * scale + jitter
            raw_coords = wrist + shift + rel  # (21, 3)

            flat = raw_coords.reshape(63)
            row: Dict[str, Any] = {"label": label, "handedness": "Right"}
            for col_name, val in zip(coord_cols, flat):
                row[col_name] = float(val)
            rows.append(row)

    return pd.DataFrame(rows)


def clean_and_validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Inspect dataset for missing target labels, NULL values, or malformed rows."""
    initial_rows = len(df)
    missing_target = int(df["label"].isna().sum()) if "label" in df.columns else initial_rows
    df_clean = df.dropna(subset=["label"]).copy()

    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    null_feature_rows = int(df_clean[coord_cols].isna().any(axis=1).sum())
    if null_feature_rows > 0:
        # Impute isolated missing coordinate cells with column median or drop if whole hand is missing
        df_clean[coord_cols] = df_clean[coord_cols].fillna(df_clean[coord_cols].median())

    valid_labels = set(STATIC_CLASSES)
    invalid_label_rows = int((~df_clean["label"].isin(valid_labels)).sum())
    df_clean = df_clean[df_clean["label"].isin(valid_labels)].reset_index(drop=True)

    diagnostics = {
        "initial_rows": initial_rows,
        "missing_target_rows_dropped": missing_target,
        "null_feature_rows_imputed": null_feature_rows,
        "invalid_label_rows_dropped": invalid_label_rows,
        "final_valid_rows": len(df_clean),
    }
    return df_clean, diagnostics


def featurize_split(df_split: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Apply wrist-centering and L2 distance normalization independently on a data split."""
    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    raw_matrix = df_split[coord_cols].to_numpy(dtype=np.float32)
    labels = df_split["label"].to_numpy()
    class_to_idx = {c: i for i, c in enumerate(STATIC_CLASSES)}

    X_norm = np.zeros((len(df_split), 63), dtype=np.float32)
    y_idx = np.zeros(len(df_split), dtype=np.int64)

    for i in range(len(df_split)):
        coords_21x3 = raw_matrix[i].reshape(21, 3)
        X_norm[i] = normalize_hand_array(coords_21x3)
        y_idx[i] = class_to_idx[str(labels[i])]

    return X_norm, y_idx


def train_and_save_demo_model(
    csv_path: Optional[Path] = None,
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    samples_per_class: int = 60,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train the 3-layer PyTorch StaticSignMLP, evaluate validation/test F1, and export checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if csv_path is not None and csv_path.exists():
        raw_df = pd.read_csv(csv_path)
    else:
        raw_df = generate_synthetic_raw_dataset(samples_per_class=samples_per_class, seed=seed)

    clean_df, data_diagnostics = clean_and_validate_dataframe(raw_df)

    # STRICT FEATURIZATION ORDERING: Split into Train (70%), Val (15%), Test (15%) BEFORE normalization
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )

    X_train, y_train = featurize_split(train_df)
    X_val, y_val = featurize_split(val_df)
    X_test, y_test = featurize_split(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticSignMLP(input_dim=63, num_classes=len(STATIC_CLASSES)).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state_dict = None

    X_val_t = torch.from_numpy(X_val).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final evaluation on Validation and held-out Test sets
    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
        X_test_t = torch.from_numpy(X_test).to(device)
        test_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()

    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_f1_macro = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, test_preds, labels=list(range(len(STATIC_CLASSES))))
    per_class_acc: Dict[str, float] = {}
    for idx, cls_name in enumerate(STATIC_CLASSES):
        row_sum = int(cm[idx].sum())
        per_class_acc[cls_name] = round(float(cm[idx, idx]) / row_sum, 4) if row_sum > 0 else 1.0

    # Export weights & TorchScript
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": STATIC_CLASSES,
            "is_custom_trained": False,
            "val_f1_macro": val_f1_macro,
            "test_f1_macro": test_f1_macro,
        },
        STATIC_WEIGHTS_PATH,
    )

    try:
        scripted = torch.jit.script(model.cpu())
        scripted.save(str(STATIC_TORCHSCRIPT_PATH))
    except Exception:
        pass

    summary: Dict[str, Any] = {
        "model_name": "StaticSignMLP (3-Layer BatchNorm + Dropout)",
        "num_classes": len(STATIC_CLASSES),
        "classes": STATIC_CLASSES,
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1_macro, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "data_diagnostics": data_diagnostics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "confusion_matrix_summary": per_class_acc,
    }

    METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        db = SignStreamDatabase()
        db.save_model_evaluation(summary)
    except Exception:
        pass

    if verbose:
        print(
            f"[SignStream Train] Completed {epochs} epochs | "
            f"Val Acc: {val_acc:.2%}, Val Macro-F1: {val_f1_macro:.2%} | "
            f"Test Acc: {test_acc:.2%}, Test Macro-F1: {test_f1_macro:.2%}"
        )
        print(f"[SignStream Train] Saved PyTorch checkpoint -> {STATIC_WEIGHTS_PATH}")

    return summary


def train_from_sqlite_dataset(
    db: Optional[SignStreamDatabase] = None,
    epochs: int = 45,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train StaticSignMLP on real user-recorded webcam samples stored in SQLite.

    Splits raw recorded samples into Train / Val / Test BEFORE applying small Gaussian
    jitter augmentation exclusively to the training split (zero data leakage).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    database = db or SignStreamDatabase()
    raw_samples = database.get_all_gesture_samples()
    if not raw_samples:
        raise ValueError("No recorded gesture samples found. Record at least 2 signs first.")

    # Group valid 63-float vectors by label
    by_label: Dict[str, List[np.ndarray]] = {}
    for label, feats in raw_samples:
        if len(feats) == 63 and all(np.isfinite(feats)):
            by_label.setdefault(label.strip().upper(), []).append(
                np.asarray(feats, dtype=np.float32)
            )

    valid_classes = sorted([lbl for lbl, vecs in by_label.items() if len(vecs) >= 5])
    if len(valid_classes) < 2:
        raise ValueError(
            "Please record at least 2 different gestures (minimum 5 samples each) before training."
        )

    class_to_idx = {c: i for i, c in enumerate(valid_classes)}

    X_train_list: List[np.ndarray] = []
    y_train_list: List[int] = []
    X_val_list: List[np.ndarray] = []
    y_val_list: List[int] = []
    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []

    # Per-class stratified split BEFORE augmentation
    for lbl in valid_classes:
        vecs = np.stack(by_label[lbl], axis=0)
        perm = rng.permutation(len(vecs))
        vecs = vecs[perm]

        n = len(vecs)
        n_test = max(1, int(round(n * 0.12)))
        n_val = max(1, int(round(n * 0.18)))
        n_train = max(1, n - n_val - n_test)

        train_vecs = vecs[:n_train]
        val_vecs = vecs[n_train : n_train + n_val]
        test_vecs = vecs[n_train + n_val :]
        if len(test_vecs) == 0:
            test_vecs = val_vecs

        idx = class_to_idx[lbl]
        for v in train_vecs:
            X_train_list.append(v)
            y_train_list.append(idx)
            # Augment training split ONLY (3 copies with realistic hand jitter)
            for _ in range(3):
                noise = rng.normal(0.0, 0.014, size=63).astype(np.float32)
                noise[:3] = 0.0  # Keep wrist origin aligned at (0, 0, 0)
                scale = float(rng.uniform(0.96, 1.04))
                aug = (v * scale + noise).astype(np.float32)
                X_train_list.append(aug)
                y_train_list.append(idx)

        for v in val_vecs:
            X_val_list.append(v)
            y_val_list.append(idx)
        for v in test_vecs:
            X_test_list.append(v)
            y_test_list.append(idx)

    X_train = np.stack(X_train_list, axis=0)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_val = np.stack(X_val_list, axis=0)
    y_val = np.array(y_val_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticSignMLP(input_dim=63, num_classes=len(valid_classes)).to(device)

    effective_batch = min(batch_size, max(4, len(X_train)))
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=len(X_train) > effective_batch and (len(X_train) % effective_batch == 1),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state_dict = None
    X_val_t = torch.from_numpy(X_val).to(device)

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            if len(bx) < 2:
                continue
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
            val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
        X_test_t = torch.from_numpy(X_test).to(device)
        test_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()

    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_f1_macro = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": valid_classes,
            "is_custom_trained": True,
            "val_f1_macro": val_f1_macro,
            "test_f1_macro": test_f1_macro,
        },
        STATIC_WEIGHTS_PATH,
    )

    summary: Dict[str, Any] = {
        "model_name": f"Custom Webcam MLP ({len(valid_classes)} Gestures)",
        "num_classes": len(valid_classes),
        "classes": valid_classes,
        "is_custom_trained": True,
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1_macro, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    database.save_model_evaluation(summary)
    return summary


def train_from_csv_dataset(
    csv_path: Path,
    epochs: int = 45,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train StaticSignMLP on any external CSV dataset (SIBI, BISINDO, ASL).

    Auto-detects either raw landmark columns (`x0,y0,z0..x20,y20,z20`) or pre-normalized
    columns (`f_0..f_62` or any 63 numeric feature columns) and splits into Train/Val/Test
    BEFORE normalization.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else ("class" if "class" in df.columns else df.columns[0])
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    raw_xyz_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    norm_f_cols = [f"f_{i}" for i in range(63)]

    if all(c in df.columns for c in raw_xyz_cols):
        feat_cols = raw_xyz_cols
        needs_normalization = True
    elif all(c in df.columns for c in norm_f_cols):
        feat_cols = norm_f_cols
        needs_normalization = False
    else:
        numeric_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 63:
            raise ValueError(
                f"CSV must contain at least 63 numeric landmark columns (found {len(numeric_cols)})."
            )
        feat_cols = numeric_cols[:63]
        needs_normalization = True

    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    # Keep classes with at least 5 samples
    counts = df[label_col].value_counts()
    valid_classes = sorted(counts[counts >= 5].index.tolist())
    if len(valid_classes) < 2:
        raise ValueError("Dataset must have at least 2 classes with >= 5 samples each.")
    df = df[df[label_col].isin(valid_classes)].reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(valid_classes)}

    # Strict featurization ordering: split BEFORE normalization
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[label_col], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[label_col], random_state=seed)

    def _to_arrays(split_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mat = split_df[feat_cols].to_numpy(dtype=np.float32)
        if needs_normalization:
            mat = np.stack([normalize_hand_array(row.reshape(21, 3)) for row in mat], axis=0)
        y = np.array([class_to_idx[c] for c in split_df[label_col]], dtype=np.int64)
        return mat, y

    X_train, y_train = _to_arrays(train_df)
    X_val, y_val = _to_arrays(val_df)
    X_test, y_test = _to_arrays(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticSignMLP(input_dim=63, num_classes=len(valid_classes)).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, max(8, len(X_train))),
        shuffle=True,
        drop_last=len(X_train) > batch_size and (len(X_train) % batch_size == 1),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state_dict = None
    X_val_t = torch.from_numpy(X_val).to(device)

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            if len(bx) < 2:
                continue
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
            val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
        test_preds = torch.argmax(model(torch.from_numpy(X_test).to(device)), dim=1).cpu().numpy()

    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_f1_macro = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": valid_classes,
            "is_custom_trained": True,
            "val_f1_macro": val_f1_macro,
            "test_f1_macro": test_f1_macro,
        },
        STATIC_WEIGHTS_PATH,
    )

    summary: Dict[str, Any] = {
        "model_name": f"External Real Dataset MLP ({len(valid_classes)} Classes)",
        "num_classes": len(valid_classes),
        "classes": valid_classes,
        "is_custom_trained": True,
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1_macro, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        SignStreamDatabase().save_model_evaluation(summary)
    except Exception:
        pass

    if verbose:
        print(
            f"[SignStream CSV Train] {len(valid_classes)} classes | "
            f"Val Acc: {val_acc:.2%}, Val Macro-F1: {val_f1_macro:.2%} | "
            f"Test Acc: {test_acc:.2%}, Test Macro-F1: {test_f1_macro:.2%}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SignStream StaticSignMLP")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV landmark dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--samples-per-class", type=int, default=60, help="Synthetic samples per class")
    parser.add_argument("--generate-demo", action="store_true", help="Train on generated anatomical dataset")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    train_and_save_demo_model(
        csv_path=csv_path,
        epochs=args.epochs,
        samples_per_class=args.samples_per_class,
        verbose=True,
    )


if __name__ == "__main__":
    main()

