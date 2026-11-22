"""Coordinate pre-processor implementing translation & scale invariance and PDD Section 7 edge-case checks.

Mathematical Formulation (PDD Section 5):
1. Translation Invariance (Origin Alignment):
   Set wrist landmark (i = 0) as the local origin (0, 0, 0):
       p'_i = p_i - p_0
2. Scale Invariance (Distance Normalization):
   Divide shifted coordinates by maximum Euclidean distance from wrist to any landmark in the same hand:
       p^norm_i = p'_i / (max_j ||p'_j||_2 + epsilon)
3. Feature Vector Shape:
   - Single Hand Static: 21 points * 3 axes = 63 float features.
   - Two-Hand Dynamic (30-frame window): 30 frames * (21 * 3 * 2 hands) = (30, 126) tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

from app.schemas import ExceptionCode, HandLandmarks, Point3D

EPSILON: float = 1e-8
OUT_OF_BOUNDS_MIN: float = 0.02
OUT_OF_BOUNDS_MAX: float = 0.98
LOW_VISIBILITY_THRESHOLD: float = 0.50
Z_DEPTH_SPIKE_STD_THRESHOLD: float = 0.25


@dataclass
class NormalizationResult:
    """Output of normalizing a single frame's hand landmarks."""

    single_hand_vector: np.ndarray  # Shape: (63,) float32
    two_hand_vector: np.ndarray     # Shape: (126,) float32
    exception_code: ExceptionCode
    exception_message: Optional[str]
    suppress_classification: bool
    wrist_xy: Tuple[float, float]


def landmarks_to_numpy(landmarks: Sequence[Point3D]) -> np.ndarray:
    """Convert 21 Point3D objects into a (21, 3) float32 numpy array."""
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(landmarks)}")
    coords = np.array([[pt.x, pt.y, pt.z] for pt in landmarks], dtype=np.float32)
    return coords


def normalize_hand_array(
    coords: np.ndarray,
    handedness: str = "Right",
    canonicalize_handedness: bool = False,
    eps: float = EPSILON,
) -> np.ndarray:
    """Normalize a (21, 3) landmark array to a (63,) translation- and scale-invariant vector.

    Args:
        coords: Raw landmark array of shape (21, 3).
        handedness: 'Left' or 'Right'.
        canonicalize_handedness: If True, mirrors X axis for Left hand so both hands share feature geometry.
        eps: Numerical stability constant.

    Returns:
        Flattened float32 array of shape (63,).
    """
    if coords.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3), got {coords.shape}")

    # Step 1: Translation Invariance (Origin Alignment to wrist i = 0)
    wrist = coords[0:1, :]  # Shape (1, 3)
    shifted = coords - wrist  # p'_i = p_i - p_0

    if canonicalize_handedness and handedness.strip().lower() == "left":
        shifted = shifted.copy()
        shifted[:, 0] = -shifted[:, 0]

    # Step 2: Scale Invariance (L2 Euclidean distance from wrist)
    distances = np.linalg.norm(shifted, axis=1)  # ||p'_j||_2 for j in 0..20
    max_distance = float(np.max(distances))

    normalized = shifted / (max_distance + eps)
    return normalized.reshape(63).astype(np.float32)


def inspect_hand_exceptions(hand: HandLandmarks) -> Tuple[ExceptionCode, Optional[str], bool]:
    """Check a detected hand against PDD Section 7 edge-case triggers.

    Returns:
        (ExceptionCode, warning_message, suppress_classification)
    """
    # 1. Low Lighting / Blur: Landmark visibility/presence confidence < 0.50
    visibilities = [
        pt.visibility if pt.visibility is not None else 1.0 for pt in hand.landmarks
    ]
    mean_visibility = float(np.mean(visibilities))
    if hand.score < LOW_VISIBILITY_THRESHOLD or mean_visibility < LOW_VISIBILITY_THRESHOLD:
        return (
            ExceptionCode.LOW_LIGHTING_BLUR,
            "Low visibility — check lighting or camera focus.",
            False,
        )

    coords = landmarks_to_numpy(hand.landmarks)
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2]

    # 2. Out-of-Bounds Hand: Any landmark coordinate < 0.02 or > 0.98 on normalized canvas axes
    if np.any(xs < OUT_OF_BOUNDS_MIN) or np.any(xs > OUT_OF_BOUNDS_MAX) or np.any(
        ys < OUT_OF_BOUNDS_MIN
    ) or np.any(ys > OUT_OF_BOUNDS_MAX):
        return (
            ExceptionCode.OUT_OF_BOUNDS,
            "Hand near edge — please center your hand within the camera frame.",
            True,
        )

    # 3. Self-Occlusion: Z-depth variance spikes or fingers overlap severely
    z_std = float(np.std(zs))
    if z_std > Z_DEPTH_SPIKE_STD_THRESHOLD:
        return (
            ExceptionCode.SELF_OCCLUSION,
            "Self-occlusion detected (Z-depth spike) — relying on temporal inertia.",
            False,
        )

    return (ExceptionCode.NONE, None, False)


def preprocess_frame_hands(hands: Sequence[HandLandmarks]) -> NormalizationResult:
    """Process 0, 1, or 2 hands from a frame into static (63,) and dynamic (126,) vectors."""
    if not hands:
        return NormalizationResult(
            single_hand_vector=np.zeros(63, dtype=np.float32),
            two_hand_vector=np.zeros(126, dtype=np.float32),
            exception_code=ExceptionCode.NO_HAND_DETECTED,
            exception_message="No hand detected — waiting for signer.",
            suppress_classification=True,
            wrist_xy=(0.5, 0.5),
        )

    primary_hand = hands[0]
    exc_code, exc_msg, suppress = inspect_hand_exceptions(primary_hand)

    primary_coords = landmarks_to_numpy(primary_hand.landmarks)
    primary_vec = normalize_hand_array(primary_coords, handedness=primary_hand.handedness)
    wrist_xy = (float(primary_coords[0, 0]), float(primary_coords[0, 1]))

    if len(hands) >= 2:
        secondary_hand = hands[1]
        sec_coords = landmarks_to_numpy(secondary_hand.landmarks)
        sec_vec = normalize_hand_array(sec_coords, handedness=secondary_hand.handedness)
    else:
        sec_vec = np.zeros(63, dtype=np.float32)

    two_hand_vec = np.concatenate([primary_vec, sec_vec], axis=0).astype(np.float32)

    return NormalizationResult(
        single_hand_vector=primary_vec,
        two_hand_vector=two_hand_vec,
        exception_code=exc_code,
        exception_message=exc_msg,
        suppress_classification=suppress,
        wrist_xy=wrist_xy,
    )

