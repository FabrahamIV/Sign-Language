"""MediaPipe Hands wrapper for server-side RGB frame landmark extraction and OpenCV dataset collection."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from app.schemas import HandLandmarks, Point3D


class MediaPipeHandExtractor:
    """Extracts up to 2 hands (21 3D keypoints per hand) from RGB images using MediaPipe Hands."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._mp_hands = None
        self._detector = None

    def _ensure_initialized(self) -> None:
        if self._detector is not None:
            return
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Server-side MediaPipe is not installed. Install with `uv pip install mediapipe opencv-python` "
                "or stream landmarks from client-side @mediapipe/tasks-vision."
            ) from exc

        self._mp_hands = mp.solutions.hands
        self._detector = self._mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def extract_from_rgb(self, rgb_frame: np.ndarray) -> List[HandLandmarks]:
        """Process an HxWx3 RGB uint8 image and return detected HandLandmarks."""
        self._ensure_initialized()
        results = self._detector.process(rgb_frame)
        hands_out: List[HandLandmarks] = []

        if not results.multi_hand_landmarks:
            return hands_out

        handedness_list = results.multi_handedness or []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            label = "Right"
            score = 0.95
            if idx < len(handedness_list) and handedness_list[idx].classification:
                cls_info = handedness_list[idx].classification[0]
                label = cls_info.label or "Right"
                score = float(cls_info.score or 0.95)

            points: List[Point3D] = []
            for lm in hand_lms.landmark:
                vis = getattr(lm, "visibility", 1.0)
                points.append(
                    Point3D(
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        visibility=float(vis) if vis is not None else 1.0,
                    )
                )

            hands_out.append(HandLandmarks(handedness=label, score=score, landmarks=points))

        return hands_out

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

