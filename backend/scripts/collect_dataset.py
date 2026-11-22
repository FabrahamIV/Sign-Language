"""OpenCV + MediaPipe interactive dataset collector (PDD Milestone 1).

Records N samples per sign class (A-Z, 0-9, IDLE, SPACE, BACKSPACE, SUBMIT), applies
wrist-centering and L2 scale normalization, and saves to CSV and/or Parquet.

Usage:
    uv run python scripts/collect_dataset.py --output data/landmarks_dataset.csv --samples 50
    uv run python scripts/collect_dataset.py --headless-synthetic --output data/landmarks_dataset.csv --samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Ensure backend root is on sys.path when invoked directly
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.model import STATIC_CLASSES, build_canonical_hand_prototypes
from app.vision.extractor import MediaPipeHandExtractor
from app.vision.normalizer import landmarks_to_numpy, normalize_hand_array


def save_dataset(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Save collected landmark rows to CSV and optionally Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
        parquet_path = output_path.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"[Collector] Also exported Parquet dataset -> {parquet_path}")
        except Exception:
            pass
    print(f"[Collector] Saved {len(df)} samples across {df['label'].nunique()} classes -> {output_path}")


def generate_headless_normalized_dataset(samples_per_class: int, output_path: Path) -> None:
    """Generate 50+ normalized samples per class (A-Z, 0-9, IDLE, SPACE, BACKSPACE, SUBMIT) without webcam."""
    rng = np.random.default_rng(42)
    prototypes = build_canonical_hand_prototypes()
    rows: List[Dict[str, Any]] = []
    feat_cols = [f"f_{i}" for i in range(63)]

    for label in STATIC_CLASSES:
        base = prototypes[label]
        for _ in range(samples_per_class):
            scale = float(rng.uniform(0.7, 1.4))
            shift = rng.uniform(-0.15, 0.15, size=(1, 3)).astype(np.float32)
            jitter = rng.normal(0.0, 0.008, size=(21, 3)).astype(np.float32)
            raw = base[0:1, :] + shift + (base - base[0:1, :]) * scale + jitter
            norm_vec = normalize_hand_array(raw)
            row: Dict[str, Any] = {"label": label, "handedness": "Right"}
            for col_name, val in zip(feat_cols, norm_vec):
                row[col_name] = float(val)
            rows.append(row)

    save_dataset(rows, output_path)


def run_interactive_opencv_collector(
    samples_per_class: int,
    output_path: Path,
    camera_index: int = 0,
) -> None:
    """Launch OpenCV window to capture and label live hand landmarks via MediaPipe."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run `uv pip install opencv-python mediapipe` "
            "or pass `--headless-synthetic`."
        ) from exc

    extractor = MediaPipeHandExtractor(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. Use `--headless-synthetic` on headless machines."
        )

    rows: List[Dict[str, Any]] = []
    feat_cols = [f"f_{i}" for i in range(63)]
    class_idx = 0
    recording = False
    collected_for_current = 0

    print("=== SignStream OpenCV Dataset Collector ===")
    print("Controls: [SPACE] Start/Pause recording current class | [N] Next class | [Q] Save & Quit")

    try:
        while class_idx < len(STATIC_CLASSES):
            target_label = STATIC_CLASSES[class_idx]
            ret, bgr_frame = cap.read()
            if not ret:
                break

            bgr_frame = cv2.flip(bgr_frame, 1)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            hands = extractor.extract_from_rgb(rgb_frame)

            status_color = (0, 255, 0) if recording else (0, 200, 255)
            cv2.putText(
                bgr_frame,
                f"Target Sign: {target_label} ({collected_for_current}/{samples_per_class})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
            )

            if hands:
                hand = hands[0]
                coords = landmarks_to_numpy(hand.landmarks)
                h, w, _ = bgr_frame.shape
                for pt in coords:
                    cx, cy = int(pt[0] * w), int(pt[1] * h)
                    cv2.circle(bgr_frame, (cx, cy), 4, (56, 189, 248), -1)

                if recording:
                    norm_vec = normalize_hand_array(coords, handedness=hand.handedness)
                    row: Dict[str, Any] = {"label": target_label, "handedness": hand.handedness}
                    for col_name, val in zip(feat_cols, norm_vec):
                        row[col_name] = float(val)
                    rows.append(row)
                    collected_for_current += 1
                    time.sleep(0.03)

                    if collected_for_current >= samples_per_class:
                        recording = False
                        collected_for_current = 0
                        class_idx += 1

            cv2.imshow("SignStream Dataset Collector", bgr_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
            elif key == ord("n"):
                recording = False
                collected_for_current = 0
                class_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    if rows:
        save_dataset(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SignStream OpenCV + MediaPipe Dataset Collector")
    parser.add_argument(
        "--output",
        type=str,
        default=str(BACKEND_ROOT / "data" / "landmarks_dataset.csv"),
        help="Output CSV or Parquet file path",
    )
    parser.add_argument("--samples", type=int, default=50, help="Samples per sign class (default: 50)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--headless-synthetic",
        action="store_true",
        help="Generate synthetic normalized dataset without opening a webcam window",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.headless_synthetic:
        generate_headless_normalized_dataset(args.samples, out_path)
    else:
        run_interactive_opencv_collector(args.samples, out_path, camera_index=args.camera)


if __name__ == "__main__":
    main()

