"""Import & convert real external Sign Language datasets (SIBI, BISINDO, ASL) into SignStream.

Supports two common dataset formats from Kaggle / HuggingFace / GitHub:
  1. Image Folder Dataset:
       dataset_dir/
         A/ img1.jpg, img2.jpg ...
         B/ img1.jpg, img2.jpg ...
         HALO/ img1.png ...
     Extracts 21x3 hand landmarks with MediaPipe Hands, applies wrist-centering + L2 scaling,
     exports to CSV, and trains StaticSignMLP.

  2. Pre-extracted Landmarks CSV:
     CSV containing a `label` (or `class`) column plus 63 landmark feature columns
     (`x0,y0,z0..x20,y20,z20` or `f_0..f_62`).

Usage:
  # From an image folder (e.g. Kaggle SIBI / ASL Alphabet / BISINDO):
  uv run python scripts/import_external_dataset.py --images ./data/sibi_images --max-per-class 300 --train

  # From a pre-extracted MediaPipe landmarks CSV:
  uv run python scripts/import_external_dataset.py --csv ./data/asl_landmarks.csv --train
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.train import train_from_csv_dataset
from app.vision.extractor import MediaPipeHandExtractor
from app.vision.normalizer import landmarks_to_numpy, normalize_hand_array

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_landmarks_from_image_folder(
    images_dir: Path,
    output_csv: Path,
    max_per_class: int = 300,
) -> Path:
    """Walk class subfolders of images, run MediaPipe Hands, and save normalized 63-d features to CSV."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV & MediaPipe are required to extract landmarks from image folders.\n"
            "Install them inside backend/.venv with:\n"
            "  uv pip install opencv-python mediapipe"
        ) from exc

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Image dataset directory not found: {images_dir}")

    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(
            f"No class subfolders found inside {images_dir}. Expected structure: dataset/A/*.jpg, dataset/B/*.jpg..."
        )

    extractor = MediaPipeHandExtractor(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    rows: List[Dict[str, Any]] = []
    feat_cols = [f"f_{i}" for i in range(63)]
    total_scanned = 0
    total_detected = 0

    try:
        for class_dir in class_dirs:
            label = class_dir.name.strip().upper().replace(" ", "_")
            img_files = [
                p
                for p in sorted(class_dir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ][:max_per_class]

            detected_for_class = 0
            for img_path in img_files:
                total_scanned += 1
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                hands = extractor.extract_from_rgb(rgb)
                if not hands:
                    continue

                hand = hands[0]
                coords = landmarks_to_numpy(hand.landmarks)
                norm_vec = normalize_hand_array(coords, handedness=hand.handedness)
                row: Dict[str, Any] = {"label": label}
                for col, val in zip(feat_cols, norm_vec):
                    row[col] = round(float(val), 6)
                rows.append(row)
                detected_for_class += 1
                total_detected += 1

            print(
                f"[Import] Class '{label}': extracted hands from {detected_for_class}/{len(img_files)} images"
            )
    finally:
        extractor.close()

    if not rows:
        raise RuntimeError(
            "MediaPipe did not detect any hands in the provided images. Check image lighting/orientation."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(
        f"[Import] Done! Saved {total_detected}/{total_scanned} hand samples across "
        f"{df['label'].nunique()} classes -> {output_csv}"
    )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a real Sign Language dataset (Image folder or Landmarks CSV) and train SignStream"
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to image folder dataset (subfolders named by sign label, e.g. A/, B/, HALO/)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a pre-extracted landmarks CSV (label + 63 coordinate columns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BACKEND_ROOT / "data" / "real_sign_dataset.csv"),
        help="Output path for extracted normalized landmarks CSV",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=300,
        help="Maximum images to process per class folder (default: 300)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=45,
        help="PyTorch training epochs if --train is enabled (default: 45)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Immediately train StaticSignMLP and overwrite app/ml/weights/static_mlp.pt",
    )
    args = parser.parse_args()

    if not args.images and not args.csv:
        parser.error("Provide either --images <folder> or --csv <file>.")

    target_csv = Path(args.csv) if args.csv else Path(args.output)
    if args.images:
        target_csv = extract_landmarks_from_image_folder(
            images_dir=Path(args.images),
            output_csv=Path(args.output),
            max_per_class=args.max_per_class,
        )

    if args.train:
        summary = train_from_csv_dataset(csv_path=target_csv, epochs=args.epochs, verbose=True)
        print(
            f"[Import] Model ready! Classes ({summary['num_classes']}): {', '.join(summary['classes'])}"
        )


if __name__ == "__main__":
    main()
