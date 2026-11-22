"""Pydantic schemas for SignStream landmark payloads, state machine outputs, and metrics."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SystemStatus(str, Enum):
    """Operational badge state for the SignStream pipeline."""

    IDLE_WAITING = "IDLE_WAITING"
    TRACKING = "TRACKING"
    LOCKED_IN = "LOCKED_IN"
    COOLDOWN = "COOLDOWN"
    WARNING = "WARNING"


class ExceptionCode(str, Enum):
    """Edge-case & exception scenario codes defined in PDD Section 7."""

    NONE = "NONE"
    NO_HAND_DETECTED = "NO_HAND_DETECTED"
    LOW_LIGHTING_BLUR = "LOW_LIGHTING_BLUR"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    TRANSITION_NOISE = "TRANSITION_NOISE"
    SELF_OCCLUSION = "SELF_OCCLUSION"


class Point3D(BaseModel):
    """Single 3D MediaPipe hand landmark coordinate."""

    x: float = Field(..., description="Normalized X coordinate [0.0, 1.0] or world X")
    y: float = Field(..., description="Normalized Y coordinate [0.0, 1.0] or world Y")
    z: float = Field(..., description="Relative Z depth from wrist")
    visibility: Optional[float] = Field(
        default=1.0, ge=0.0, le=1.0, description="Landmark presence/visibility score"
    )


class HandLandmarks(BaseModel):
    """21 3D landmarks representing a single detected hand."""

    handedness: str = Field(default="Right", description="'Left' or 'Right' hand label")
    score: float = Field(
        default=0.98, ge=0.0, le=1.0, description="MediaPipe hand detection confidence"
    )
    landmarks: List[Point3D] = Field(
        ...,
        min_length=21,
        max_length=21,
        description="Exactly 21 3D hand keypoints ordered 0 (WRIST) to 20 (PINKY_TIP)",
    )


class FramePayload(BaseModel):
    """Incoming WebSocket frame payload from client-side MediaPipe or simulator."""

    frame_id: int = Field(default=0, ge=0, description="Monotonically increasing frame counter")
    timestamp_ms: float = Field(..., description="Client capture timestamp in milliseconds")
    hands: List[HandLandmarks] = Field(
        default_factory=list,
        max_length=2,
        description="0, 1, or 2 detected hands in the current frame",
    )
    hint_label: Optional[str] = Field(
        default=None,
        description="Optional ground-truth or simulator gesture hint when recording/testing",
    )
    command: Optional[str] = Field(
        default=None,
        description="Optional explicit client control action ('CLEAR', 'BACKSPACE', 'SPACE', 'SUBMIT', 'REFINE')",
    )

    @field_validator("hands")
    @classmethod
    def validate_hand_count(cls, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        if len(hands) > 2:
            return hands[:2]
        return hands


class TopPrediction(BaseModel):
    """Class prediction entry for top-K confidence visualization."""

    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    """Real-time response returned over WebSocket for every processed frame."""

    frame_id: int
    client_timestamp_ms: float
    inference_latency_ms: float
    status: SystemStatus
    exception_code: ExceptionCode = ExceptionCode.NONE
    exception_message: Optional[str] = None

    # Classification details
    raw_prediction: Optional[str] = None
    confidence: float = 0.0
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    dynamic_token: Optional[str] = None

    # State machine / aggregator details
    locked_character: Optional[str] = None
    emitted_event: Optional[str] = None
    window_fill: int = 0
    window_size: int = 15
    agreement_count: int = 0
    lock_threshold: int = 12
    cooldown_remaining: int = 0
    open_palm_hold_progress: float = 0.0

    # Text buffers
    active_text: str = ""
    refined_sentence: str = ""
    submitted_sentences: List[str] = Field(default_factory=list)

    # Normalized feature vector preview (first 6 floats for diagnostics)
    normalized_preview: List[float] = Field(default_factory=list)


class RecordSampleRequest(BaseModel):
    """Payload to record a labeled static hand landmark sample into the dataset."""

    label: str = Field(..., min_length=1, max_length=32)
    hand: HandLandmarks


class RecordBatchRequest(BaseModel):
    """Payload to burst-record multiple labeled 21-landmark hand frames in one request."""

    label: str = Field(..., min_length=1, max_length=32)
    hands: List[HandLandmarks] = Field(..., min_length=1, max_length=200)


class TrainModelRequest(BaseModel):
    """Request parameters for training the PyTorch MLP on user-recorded webcam samples."""

    epochs: int = Field(default=45, ge=10, le=200)


class GrammarRefineRequest(BaseModel):
    """Request payload for post-processing raw sign glosses / characters into natural text."""

    raw_text: str
    language: str = Field(default="ID", description="'ID' (SIBI/BISINDO) or 'EN' (ASL)")


class GrammarRefineResponse(BaseModel):
    """Refined grammatical output from raw character/gloss sequence."""

    raw_text: str
    refined_text: str
    corrections_applied: List[str] = Field(default_factory=list)


class ModelMetricsSummary(BaseModel):
    """Saved training & evaluation metrics for the PyTorch classifier."""

    model_name: str
    num_classes: int
    classes: List[str]
    val_accuracy: float
    val_f1_macro: float
    test_accuracy: float
    test_f1_macro: float
    train_samples: int
    val_samples: int
    test_samples: int
    trained_at: str
    confusion_matrix_summary: Dict[str, float] = Field(default_factory=dict)

