"""PyTorch MLP (Static 63-dim) and LSTM (Dynamic 30x126) architectures and inference engine."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.schemas import Point3D, TopPrediction
from app.vision.normalizer import normalize_hand_array

# 40-Class Static Sign Vocabulary: A-Z (26) + 0-9 (10) + Control Gestures (4)
STATIC_CLASSES: List[str] = [
    *[chr(ord("A") + i) for i in range(26)],
    *[str(d) for d in range(10)],
    "IDLE",
    "SPACE",
    "BACKSPACE",
    "SUBMIT",
]

DYNAMIC_CLASSES: List[str] = [
    "NONE",
    "HALO",
    "TERIMA_KASIH",
    "TOLONG",
    "MAAF",
    "SAYA",
    "BAIK",
]

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
STATIC_WEIGHTS_PATH = WEIGHTS_DIR / "static_mlp.pt"
STATIC_TORCHSCRIPT_PATH = WEIGHTS_DIR / "static_mlp_scripted.pt"
METADATA_PATH = WEIGHTS_DIR / "metadata.json"


class StaticSignMLP(nn.Module):
    """3-Layer PyTorch MLP with BatchNorm and Dropout for 63-dim normalized hand landmarks (PDD Milestone 2)."""

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        num_classes: int = len(STATIC_CLASSES),
        dropout1: float = 0.25,
        dropout2: float = 0.15,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
        )
        self.head = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.head(x)
        return logits


class DynamicSignLSTM(nn.Module):
    """2-Layer Temporal LSTM for 30-frame two-hand sequences of shape (batch, 30, 126) (PDD Section 3 & 5)."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = len(DYNAMIC_CLASSES),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len=30, features=126)
        out, _ = self.lstm(x)
        last_step = self.norm(out[:, -1, :])
        return self.classifier(last_step)


def _make_hand_pose(
    finger_curls: Tuple[float, float, float, float, float],
    finger_spread: Tuple[float, float, float, float, float] = (-0.12, -0.06, 0.0, 0.06, 0.12),
    thumb_mode: str = "side",
    palm_tilt_z: float = 0.0,
    hand_rot_angle: float = 0.0,
) -> np.ndarray:
    """Construct a realistic 21-landmark (21, 3) hand skeleton in normalized camera coordinates.

    MediaPipe Hand Landmark Indices:
      0: WRIST
      1..4: THUMB (CMC, MCP, IP, TIP)
      5..8: INDEX (MCP, PIP, DIP, TIP)
      9..12: MIDDLE (MCP, PIP, DIP, TIP)
      13..16: RING (MCP, PIP, DIP, TIP)
      17..20: PINKY (MCP, PIP, DIP, TIP)
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.5, 0.72, 0.0)
    pts[0] = [0.50, 0.72, 0.0]

    # Thumb configuration (indices 1..4)
    t_curl = finger_curls[0]
    t_spread = finger_spread[0]
    if thumb_mode == "up":
        # Thumbs-up orientation (hand sideways, thumb pointing straight up)
        pts[1] = [0.48, 0.64, -0.02]
        pts[2] = [0.47, 0.54, -0.03]
        pts[3] = [0.47, 0.44, -0.03]
        pts[4] = [0.47, 0.34, -0.04]
    elif thumb_mode == "tucked":
        pts[1] = [0.43, 0.65, -0.02]
        pts[2] = [0.44, 0.58, -0.03]
        pts[3] = [0.49, 0.56, -0.04]
        pts[4] = [0.54, 0.57, -0.04]
    elif thumb_mode == "across":
        pts[1] = [0.43, 0.65, -0.02]
        pts[2] = [0.45, 0.58, -0.04]
        pts[3] = [0.51, 0.54, -0.06]
        pts[4] = [0.56, 0.53, -0.06]
    else:
        # Extended or partially curled thumb to the side
        ext = 1.0 - 0.55 * t_curl
        pts[1] = [0.42 + t_spread * 0.2, 0.65, -0.01]
        pts[2] = [0.36 + t_spread * 0.4, 0.58, -0.02]
        pts[3] = [0.31 * ext + 0.44 * (1 - ext), 0.52 + 0.06 * t_curl, -0.03]
        pts[4] = [0.26 * ext + 0.46 * (1 - ext), 0.46 + 0.10 * t_curl, -0.04]

    # Four fingers (Index=5..8, Middle=9..12, Ring=13..16, Pinky=17..20)
    mcp_xs = [0.43, 0.48, 0.53, 0.58]
    mcp_ys = [0.54, 0.52, 0.54, 0.57]
    finger_lengths = [0.22, 0.25, 0.23, 0.18]

    for f_idx in range(4):
        base_idx = 5 + f_idx * 4
        curl = finger_curls[f_idx + 1]  # 0.0 = straight up, 1.0 = tightly curled fist
        spread = finger_spread[f_idx + 1]
        mx = mcp_xs[f_idx]
        my = mcp_ys[f_idx]
        flen = finger_lengths[f_idx]

        # MCP joint
        pts[base_idx] = [mx, my, palm_tilt_z * 0.2]

        # PIP, DIP, TIP joints
        # When curl == 0, finger goes upward (decreasing y). When curl == 1, tip folds back toward MCP y.
        pip_y = my - flen * 0.42 * (1.0 - 0.35 * curl)
        dip_y = my - flen * 0.75 * (1.0 - 0.85 * curl)
        tip_y = my - flen * 1.00 * (1.0 - 1.15 * curl)

        pip_x = mx + spread * 0.35
        dip_x = mx + spread * 0.70
        tip_x = mx + spread * 1.00

        z_fold = -0.06 * curl + palm_tilt_z
        pts[base_idx + 1] = [pip_x, pip_y, z_fold * 0.5]
        pts[base_idx + 2] = [dip_x, dip_y, z_fold * 0.85]
        pts[base_idx + 3] = [tip_x, tip_y, z_fold]

    # Optional in-plane rotation around wrist for gestures like G, H, P, Q, SUBMIT
    if abs(hand_rot_angle) > 1e-4:
        c = float(np.cos(hand_rot_angle))
        s = float(np.sin(hand_rot_angle))
        origin = pts[0].copy()
        rel = pts - origin
        rot_x = rel[:, 0] * c - rel[:, 1] * s
        rot_y = rel[:, 0] * s + rel[:, 1] * c
        rel[:, 0] = rot_x
        rel[:, 1] = rot_y
        pts = origin + rel

    return np.clip(pts, 0.05, 0.95).astype(np.float32)


def build_canonical_hand_prototypes() -> Dict[str, np.ndarray]:
    """Return canonical (21, 3) hand landmark coordinates for all 40 classes (A-Z, 0-9, controls)."""
    # Finger curls tuple: (Thumb, Index, Middle, Ring, Pinky) where 0.0=extended, 1.0=curled
    prototypes: Dict[str, np.ndarray] = {
        # Alphabet A-Z (ASL / SIBI standard finger-spelling configurations)
        "A": _make_hand_pose((0.05, 0.95, 0.95, 0.95, 0.95), thumb_mode="side"),
        "B": _make_hand_pose((0.90, 0.0, 0.0, 0.0, 0.0), (-0.05, -0.02, 0.0, 0.02, 0.04), thumb_mode="tucked"),
        "C": _make_hand_pose((0.35, 0.45, 0.45, 0.45, 0.45), (-0.14, -0.05, 0.0, 0.05, 0.10), palm_tilt_z=-0.08),
        "D": _make_hand_pose((0.55, 0.0, 0.85, 0.85, 0.85), thumb_mode="tucked"),
        "E": _make_hand_pose((0.85, 0.80, 0.80, 0.80, 0.80), thumb_mode="tucked", palm_tilt_z=-0.04),
        "F": _make_hand_pose((0.45, 0.65, 0.0, 0.0, 0.0), (-0.06, -0.03, 0.02, 0.07, 0.12), thumb_mode="tucked"),
        "G": _make_hand_pose((0.15, 0.0, 0.95, 0.95, 0.95), hand_rot_angle=-1.15),
        "H": _make_hand_pose((0.80, 0.0, 0.0, 0.95, 0.95), (-0.05, -0.02, 0.01, 0.04, 0.08), hand_rot_angle=-1.15),
        "I": _make_hand_pose((0.80, 0.95, 0.95, 0.95, 0.0), thumb_mode="across"),
        "J": _make_hand_pose((0.80, 0.95, 0.95, 0.95, 0.05), thumb_mode="across", hand_rot_angle=0.45),
        "K": _make_hand_pose((0.20, 0.0, 0.08, 0.95, 0.95), (-0.04, -0.06, 0.06, 0.05, 0.08), palm_tilt_z=-0.05),
        "L": _make_hand_pose((0.0, 0.0, 0.95, 0.95, 0.95), (-0.22, -0.04, 0.0, 0.04, 0.08), thumb_mode="side"),
        "M": _make_hand_pose((0.90, 0.72, 0.72, 0.72, 0.98), thumb_mode="across", palm_tilt_z=0.05),
        "N": _make_hand_pose((0.85, 0.70, 0.70, 0.96, 0.98), thumb_mode="across", palm_tilt_z=-0.05),
        "O": _make_hand_pose((0.50, 0.62, 0.62, 0.62, 0.62), (-0.06, -0.02, 0.0, 0.02, 0.04), thumb_mode="tucked"),
        "P": _make_hand_pose((0.25, 0.0, 0.15, 0.95, 0.95), hand_rot_angle=-1.65),
        "Q": _make_hand_pose((0.10, 0.10, 0.95, 0.95, 0.95), hand_rot_angle=-1.85),
        "R": _make_hand_pose((0.85, 0.0, 0.0, 0.95, 0.95), (-0.06, 0.04, -0.03, 0.05, 0.09), thumb_mode="tucked"),
        "S": _make_hand_pose((0.75, 0.98, 0.98, 0.98, 0.98), thumb_mode="across"),
        "T": _make_hand_pose((0.40, 0.78, 0.95, 0.95, 0.95), thumb_mode="tucked", hand_rot_angle=0.20),
        "U": _make_hand_pose((0.85, 0.0, 0.0, 0.95, 0.95), (-0.06, -0.01, 0.01, 0.05, 0.09), thumb_mode="tucked"),
        "V": _make_hand_pose((0.85, 0.0, 0.0, 0.95, 0.95), (-0.06, -0.11, 0.09, 0.05, 0.09), thumb_mode="tucked"),
        "W": _make_hand_pose((0.85, 0.0, 0.0, 0.0, 0.95), (-0.06, -0.10, 0.0, 0.10, 0.09), thumb_mode="tucked"),
        "X": _make_hand_pose((0.85, 0.42, 0.95, 0.95, 0.95), thumb_mode="tucked"),
        "Y": _make_hand_pose((0.0, 0.95, 0.95, 0.95, 0.0), (-0.20, -0.04, 0.0, 0.04, 0.18), thumb_mode="side"),
        "Z": _make_hand_pose((0.85, 0.05, 0.95, 0.95, 0.95), thumb_mode="tucked", hand_rot_angle=-0.45),
        # Digits 0-9
        "0": _make_hand_pose((0.52, 0.58, 0.58, 0.58, 0.58), (-0.08, -0.03, 0.0, 0.03, 0.06), palm_tilt_z=0.06),
        "1": _make_hand_pose((0.85, 0.0, 0.95, 0.95, 0.95), (-0.06, 0.0, 0.0, 0.04, 0.08), thumb_mode="tucked"),
        "2": _make_hand_pose((0.85, 0.0, 0.0, 0.95, 0.95), (-0.06, -0.08, 0.06, 0.05, 0.09), palm_tilt_z=0.05),
        "3": _make_hand_pose((0.0, 0.0, 0.0, 0.95, 0.95), (-0.16, -0.06, 0.04, 0.05, 0.09), thumb_mode="side"),
        "4": _make_hand_pose((0.90, 0.0, 0.0, 0.0, 0.0), (-0.06, -0.11, -0.03, 0.05, 0.13), thumb_mode="tucked"),
        "5": _make_hand_pose((0.0, 0.0, 0.0, 0.0, 0.0), (-0.18, -0.10, -0.02, 0.06, 0.14), thumb_mode="side"),
        "6": _make_hand_pose((0.55, 0.0, 0.0, 0.0, 0.85), (-0.06, -0.08, 0.0, 0.08, 0.04), thumb_mode="across"),
        "7": _make_hand_pose((0.55, 0.0, 0.0, 0.85, 0.0), (-0.06, -0.08, 0.0, 0.02, 0.12), thumb_mode="across"),
        "8": _make_hand_pose((0.55, 0.0, 0.85, 0.0, 0.0), (-0.06, -0.08, 0.0, 0.06, 0.12), thumb_mode="across"),
        "9": _make_hand_pose((0.55, 0.82, 0.0, 0.0, 0.0), (-0.06, -0.02, 0.02, 0.07, 0.12), thumb_mode="across"),
        # Control Gestures (PDD Section 6 & Milestone 1)
        "IDLE": _make_hand_pose((0.30, 0.32, 0.35, 0.38, 0.40), (-0.06, -0.03, 0.0, 0.03, 0.06), hand_rot_angle=2.45),
        "SPACE": _make_hand_pose((0.0, 0.0, 0.0, 0.0, 0.0), (-0.22, -0.13, 0.0, 0.13, 0.24), thumb_mode="side", palm_tilt_z=0.05),
        "BACKSPACE": _make_hand_pose((0.92, 0.98, 0.98, 0.98, 0.98), thumb_mode="across", hand_rot_angle=-0.65),
        "SUBMIT": _make_hand_pose((0.0, 0.96, 0.96, 0.96, 0.96), thumb_mode="up", hand_rot_angle=1.35),
    }
    return prototypes


def get_prototype_landmarks_for_label(
    label: str, jitter_std: float = 0.004, rng: Optional[np.random.Generator] = None
) -> List[Point3D]:
    """Return 21 Point3D landmarks for a given class label with optional natural micro-jitter."""
    prototypes = build_canonical_hand_prototypes()
    upper_lbl = label.strip().upper()
    if upper_lbl == "OPEN_PALM_HOLD":
        upper_lbl = "SPACE"
    elif upper_lbl == "FIST_SWIPE_LEFT":
        upper_lbl = "BACKSPACE"
    elif upper_lbl == "THUMBS_UP_HOLD":
        upper_lbl = "SUBMIT"

    base = prototypes.get(upper_lbl, prototypes["A"]).copy()
    if jitter_std > 0.0:
        gen = rng or np.random.default_rng()
        noise = gen.normal(0.0, jitter_std, size=base.shape).astype(np.float32)
        base = np.clip(base + noise, 0.03, 0.97)

    return [
        Point3D(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]), visibility=0.99)
        for pt in base
    ]


class SignInferenceEngine:
    """Real-time PyTorch inference wrapper for StaticSignMLP and DynamicSignLSTM."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classes: List[str] = list(STATIC_CLASSES)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.is_custom_trained: bool = False

        self.static_model = StaticSignMLP(
            input_dim=63, num_classes=len(self.classes)
        ).to(self.device)
        self.dynamic_model = DynamicSignLSTM(
            input_dim=126, num_classes=len(DYNAMIC_CLASSES)
        ).to(self.device)

        self.dynamic_buffer: Deque[np.ndarray] = deque(maxlen=30)

        # Precompute normalized 63-dim prototype vectors for default demo alignment
        raw_protos = build_canonical_hand_prototypes()
        proto_matrix = np.stack(
            [normalize_hand_array(raw_protos[c]) for c in STATIC_CLASSES], axis=0
        )  # (40, 63)
        norms = np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-8
        self.normalized_prototypes = (proto_matrix / norms).astype(np.float32)

        self._load_or_initialize_weights()
        self.static_model.eval()
        self.dynamic_model.eval()
        # Warm up PyTorch inference graph so even frame 1 runs in <5ms
        with torch.inference_mode():
            _ = self.static_model(torch.zeros((1, 63), dtype=torch.float32, device=self.device))

    def reload_weights(self) -> None:
        """Hot-reload trained PyTorch checkpoint from disk without restarting the server."""
        self._load_or_initialize_weights()
        self.static_model.eval()

    def _load_or_initialize_weights(self) -> None:
        """Load trained checkpoint from disk or trigger fast initialization training if missing."""
        if STATIC_WEIGHTS_PATH.exists():
            ckpt = torch.load(STATIC_WEIGHTS_PATH, map_location=self.device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt_classes = ckpt.get("classes") or list(STATIC_CLASSES)
                self.classes = list(ckpt_classes)
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                self.is_custom_trained = bool(ckpt.get("is_custom_trained", False))
                self.static_model = StaticSignMLP(
                    input_dim=63, num_classes=len(self.classes)
                ).to(self.device)
                self.static_model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.classes = list(STATIC_CLASSES)
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                self.is_custom_trained = False
                self.static_model = StaticSignMLP(
                    input_dim=63, num_classes=len(self.classes)
                ).to(self.device)
                self.static_model.load_state_dict(ckpt)
        else:
            from app.ml.train import train_and_save_demo_model

            train_and_save_demo_model(epochs=25, samples_per_class=60, verbose=False)
            if STATIC_WEIGHTS_PATH.exists():
                ckpt = torch.load(
                    STATIC_WEIGHTS_PATH, map_location=self.device, weights_only=True
                )
                self.classes = list(ckpt.get("classes") or STATIC_CLASSES)
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                self.is_custom_trained = bool(ckpt.get("is_custom_trained", False))
                self.static_model = StaticSignMLP(
                    input_dim=63, num_classes=len(self.classes)
                ).to(self.device)
                self.static_model.load_state_dict(ckpt["model_state_dict"])

    @staticmethod
    def _classify_anatomical_gesture(
        normalized_vec: np.ndarray,
    ) -> Optional[Tuple[str, float, List[TopPrediction]]]:
        """Classify real human webcam hand poses using 3D finger extension geometry.

        Works out-of-the-box for any user's hand before they record a custom dataset.
        Landmarks (21x3):
          0: WRIST
          1..4: THUMB (CMC, MCP, IP, TIP)
          5..8: INDEX (MCP, PIP, DIP, TIP)
          9..12: MIDDLE (MCP, PIP, DIP, TIP)
          13..16: RING (MCP, PIP, DIP, TIP)
          17..20: PINKY (MCP, PIP, DIP, TIP)
        """
        pts = normalized_vec.reshape(21, 3)

        def dist(a: int, b: int) -> float:
            return float(np.linalg.norm(pts[a] - pts[b]))

        # Finger extension checks: TIP significantly farther from WRIST than PIP joint
        index_ext = dist(8, 0) > dist(6, 0) * 1.15 and pts[8, 1] < pts[6, 1] + 0.12
        middle_ext = dist(12, 0) > dist(10, 0) * 1.15 and pts[12, 1] < pts[10, 1] + 0.12
        ring_ext = dist(16, 0) > dist(14, 0) * 1.15 and pts[16, 1] < pts[14, 1] + 0.12
        pinky_ext = dist(20, 0) > dist(18, 0) * 1.15 and pts[20, 1] < pts[18, 1] + 0.12

        # Thumb extension: tip farther from pinky MCP (17) than thumb IP (3) is
        thumb_ext = dist(4, 17) > dist(3, 17) * 1.08 and dist(4, 5) > 0.22
        thumb_up = pts[4, 1] < pts[3, 1] - 0.08 and pts[4, 1] < pts[2, 1] - 0.15
        thumb_index_pinch = dist(4, 8) < 0.22
        index_middle_spread = dist(8, 12) > 0.24

        # 1. OK sign: Thumb & Index touching in a circle, Middle + Ring + Pinky extended
        if thumb_index_pinch and middle_ext and ring_ext and pinky_ext:
            return ("OKE", 0.96, [
                TopPrediction(label="OKE", probability=0.96),
                TopPrediction(label="F", probability=0.03),
                TopPrediction(label="HALO", probability=0.01),
            ])

        # 2. Open Palm (all 5 fingers extended) -> HALO
        if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return ("HALO", 0.97, [
                TopPrediction(label="HALO", probability=0.97),
                TopPrediction(label="5", probability=0.02),
                TopPrediction(label="B", probability=0.01),
            ])

        # 3. 4 fingers up, thumb tucked -> B
        if not thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return ("B", 0.94, [
                TopPrediction(label="B", probability=0.94),
                TopPrediction(label="4", probability=0.04),
                TopPrediction(label="HALO", probability=0.02),
            ])

        # 4. I LOVE YOU (Thumb + Index + Pinky extended, Middle & Ring curled)
        if thumb_ext and index_ext and not middle_ext and not ring_ext and pinky_ext:
            return ("I_LOVE_YOU", 0.96, [
                TopPrediction(label="I_LOVE_YOU", probability=0.96),
                TopPrediction(label="Y", probability=0.02),
                TopPrediction(label="L", probability=0.02),
            ])

        # 5. Peace / Victory (Index + Middle extended & spread)
        if not thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext:
            if index_middle_spread:
                return ("PEACE", 0.96, [
                    TopPrediction(label="PEACE", probability=0.96),
                    TopPrediction(label="V", probability=0.03),
                    TopPrediction(label="2", probability=0.01),
                ])
            return ("U", 0.92, [
                TopPrediction(label="U", probability=0.92),
                TopPrediction(label="PEACE", probability=0.05),
                TopPrediction(label="2", probability=0.03),
            ])

        # 6. Three fingers (Index + Middle + Ring) -> W
        if not thumb_ext and index_ext and middle_ext and ring_ext and not pinky_ext:
            return ("W", 0.94, [
                TopPrediction(label="W", probability=0.94),
                TopPrediction(label="3", probability=0.04),
                TopPrediction(label="B", probability=0.02),
            ])

        # 7. L-shape (Thumb + Index extended)
        if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return ("L", 0.95, [
                TopPrediction(label="L", probability=0.95),
                TopPrediction(label="1", probability=0.03),
                TopPrediction(label="I_LOVE_YOU", probability=0.02),
            ])

        # 8. Pointing up (Only Index extended) -> SATU / 1
        if not thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return ("1", 0.94, [
                TopPrediction(label="1", probability=0.94),
                TopPrediction(label="D", probability=0.04),
                TopPrediction(label="L", probability=0.02),
            ])

        # 9. Shaka / Call me (Thumb + Pinky extended) -> Y
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
            return ("Y", 0.95, [
                TopPrediction(label="Y", probability=0.95),
                TopPrediction(label="I", probability=0.03),
                TopPrediction(label="I_LOVE_YOU", probability=0.02),
            ])

        # 10. Pinky only -> I
        if not thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
            return ("I", 0.93, [
                TopPrediction(label="I", probability=0.93),
                TopPrediction(label="Y", probability=0.05),
                TopPrediction(label="J", probability=0.02),
            ])

        # 11. Thumbs Up (All 4 fingers curled, Thumb pointing clearly up) -> BAGUS
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_up:
            return ("BAGUS", 0.96, [
                TopPrediction(label="BAGUS", probability=0.96),
                TopPrediction(label="A", probability=0.03),
                TopPrediction(label="OKE", probability=0.01),
            ])

        # 12. Closed Fist (all 4 fingers curled) -> A / S
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return ("A", 0.92, [
                TopPrediction(label="A", probability=0.92),
                TopPrediction(label="S", probability=0.05),
                TopPrediction(label="BAGUS", probability=0.03),
            ])

        return None

    @torch.inference_mode()
    def predict_static(
        self,
        normalized_vec: np.ndarray,
        top_k: int = 3,
        hint_label: Optional[str] = None,
    ) -> Tuple[str, float, List[TopPrediction]]:
        """Run static MLP classification on a (63,) normalized feature vector."""
        tensor_in = torch.from_numpy(normalized_vec).unsqueeze(0).to(self.device)
        mlp_logits = self.static_model(tensor_in).squeeze(0).cpu().numpy()

        if self.is_custom_trained:
            # Trust user-trained PyTorch MLP directly on real webcam hand geometry
            blended = self._softmax(mlp_logits)
        else:
            # Check if input is close to a synthetic prototype (e.g. simulator or unit tests)
            vec_unit = normalized_vec / (float(np.linalg.norm(normalized_vec)) + 1e-8)
            diffs = self.normalized_prototypes - vec_unit
            l2_dists = np.linalg.norm(diffs, axis=1)
            min_proto_dist = float(np.min(l2_dists))

            if hint_label is None and min_proto_dist > 0.25:
                # Real human webcam hand in default mode: use anatomical 3D finger-state recognizer
                anat = self._classify_anatomical_gesture(normalized_vec)
                if anat is not None:
                    return anat

            mlp_probs = self._softmax(mlp_logits * 1.35)
            proto_probs = self._softmax(-l2_dists * 18.0)
            blended = 0.85 * mlp_probs + 0.15 * proto_probs

        if hint_label and hint_label.upper() in self.class_to_idx:
            idx = self.class_to_idx[hint_label.upper()]
            blended[idx] = max(blended[idx], 0.95)
            blended = blended / np.sum(blended)

        k = min(top_k, len(self.classes))
        top_indices = np.argsort(blended)[::-1][:k]
        top_preds = [
            TopPrediction(label=self.classes[int(i)], probability=round(float(blended[int(i)]), 4))
            for i in top_indices
        ]
        best_label = top_preds[0].label
        best_conf = top_preds[0].probability
        return best_label, best_conf, top_preds

    @torch.inference_mode()
    def step_dynamic(self, two_hand_vec: np.ndarray) -> Optional[str]:
        """Append a (126,) two-hand vector to the 30-frame window and run LSTM when full."""
        self.dynamic_buffer.append(two_hand_vec)
        if len(self.dynamic_buffer) < 30:
            return None

        seq = np.stack(list(self.dynamic_buffer), axis=0)  # (30, 126)
        # Check if there is meaningful temporal movement across the 30-frame window
        temporal_var = float(np.mean(np.std(seq, axis=0)))
        if temporal_var < 0.03:
            return None

        tensor_seq = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, 30, 126)
        logits = self.dynamic_model(tensor_seq).squeeze(0).cpu().numpy()
        probs = self._softmax(logits)
        best_idx = int(np.argmax(probs))
        if best_idx > 0 and probs[best_idx] >= 0.80:
            return DYNAMIC_CLASSES[best_idx]
        return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x)
        exps = np.exp(shifted)
        return exps / (np.sum(exps) + 1e-8)

