"""FastAPI application entry point and low-latency WebSocket streaming server for SignStream."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import time
from typing import Any, Dict, Optional
import uuid
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.database import SignStreamDatabase
from app.ml.model import (
    DYNAMIC_CLASSES,
    METADATA_PATH,
    STATIC_CLASSES,
    SignInferenceEngine,
    build_canonical_hand_prototypes,
    get_prototype_landmarks_for_label,
)
from app.ml.train import train_and_save_demo_model, train_from_sqlite_dataset
from app.schemas import (
    ExceptionCode,
    FramePayload,
    GrammarRefineRequest,
    GrammarRefineResponse,
    InferenceResponse,
    RecordBatchRequest,
    RecordSampleRequest,
    SystemStatus,
    TrainModelRequest,
)
from app.vision.aggregator import SignStateAggregator, refine_gloss_sentence
from app.vision.normalizer import landmarks_to_numpy, normalize_hand_array, preprocess_frame_hands


class AppState:
    """Singleton container for the PyTorch inference engine and SQLite database."""

    engine: Optional[SignInferenceEngine] = None
    db: Optional[SignStreamDatabase] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database and PyTorch inference engine on startup."""
    state.db = SignStreamDatabase()
    state.engine = SignInferenceEngine()
    yield


app = FastAPI(
    title="SignStream — Real-Time Sign Language Translation API",
    version="1.0.0",
    description=(
        "Low-latency WebSocket & REST API for 3D hand landmark normalization, "
        "PyTorch MLP/LSTM gesture classification, and deterministic state aggregation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_engine() -> SignInferenceEngine:
    if state.engine is None:
        state.engine = SignInferenceEngine()
    return state.engine


def _get_db() -> SignStreamDatabase:
    if state.db is None:
        state.db = SignStreamDatabase()
    return state.db


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Return service health and PyTorch model status."""
    engine = _get_engine()
    return {
        "status": "healthy",
        "service": "signstream-backend",
        "version": "1.0.0",
        "device": str(engine.device),
        "num_static_classes": len(STATIC_CLASSES),
        "num_dynamic_classes": len(DYNAMIC_CLASSES),
    }


@app.get("/api/classes")
async def get_vocabulary_and_prototypes() -> Dict[str, Any]:
    """Return supported static/dynamic classes and canonical 21-point 3D hand prototypes."""
    raw_protos = build_canonical_hand_prototypes()
    serialized_protos: Dict[str, Any] = {}
    for label, coords in raw_protos.items():
        serialized_protos[label] = [
            {"x": round(float(pt[0]), 4), "y": round(float(pt[1]), 4), "z": round(float(pt[2]), 4), "visibility": 0.99}
            for pt in coords
        ]
    return {
        "static_classes": STATIC_CLASSES,
        "dynamic_classes": DYNAMIC_CLASSES,
        "control_gestures": {
            "OPEN_PALM_HOLD": "Insert Space (' ') after 1.5s hold or SPACE lock-in",
            "FIST_SWIPE_LEFT": "Backspace (delete last character)",
            "THUMBS_UP_HOLD": "Submit / clear active sentence buffer",
            "IDLE": "Neutral hand pose (resets repeat-prevention cooldown after 10 frames)",
        },
        "prototypes": serialized_protos,
    }


@app.get("/api/metrics")
async def get_model_and_session_metrics() -> Dict[str, Any]:
    """Return latest PyTorch model evaluation metrics and recent SQLite session logs."""
    db = _get_db()
    eval_summary = db.get_latest_evaluation()
    if eval_summary is None and METADATA_PATH.exists():
        eval_summary = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return {
        "model_evaluation": eval_summary,
        "sample_counts": db.get_sample_counts(),
        "recent_emissions": db.get_recent_logs(limit=20),
    }


@app.get("/api/samples")
async def get_recorded_samples_summary() -> Dict[str, Any]:
    """Return recorded sample counts per gesture label and active model state."""
    db = _get_db()
    engine = _get_engine()
    counts = db.get_sample_counts()
    return {
        "sample_counts": counts,
        "total_samples": sum(counts.values()),
        "num_recorded_classes": len(counts),
        "active_classes": engine.classes,
        "is_custom_trained": engine.is_custom_trained,
    }


@app.post("/api/samples")
async def record_gesture_sample(req: RecordSampleRequest) -> Dict[str, Any]:
    """Normalize and store a labeled 21-landmark hand sample in SQLite."""
    db = _get_db()
    coords = landmarks_to_numpy(req.hand.landmarks)
    norm_vec = normalize_hand_array(coords, handedness=req.hand.handedness)
    row_id = db.insert_gesture_sample(
        label=req.label,
        handedness=req.hand.handedness,
        normalized_features=[round(float(v), 6) for v in norm_vec.tolist()],
    )
    return {
        "saved": True,
        "sample_id": row_id,
        "label": req.label.strip().upper(),
        "feature_dim": len(norm_vec),
        "sample_counts": db.get_sample_counts(),
    }


@app.post("/api/samples/batch")
async def record_gesture_samples_batch(req: RecordBatchRequest) -> Dict[str, Any]:
    """Normalize and store a burst of labeled 21-landmark hand frames in SQLite."""
    db = _get_db()
    normalized_batch = []
    for hand in req.hands:
        coords = landmarks_to_numpy(hand.landmarks)
        norm_vec = normalize_hand_array(coords, handedness=hand.handedness)
        normalized_batch.append(
            (hand.handedness, [round(float(v), 6) for v in norm_vec.tolist()])
        )
    inserted = db.insert_gesture_samples_batch(req.label, normalized_batch)
    return {
        "saved": True,
        "inserted_count": inserted,
        "label": req.label.strip().upper(),
        "sample_counts": db.get_sample_counts(),
    }


@app.delete("/api/samples/{label}")
async def delete_recorded_samples(label: str) -> Dict[str, Any]:
    """Delete recorded samples for a specific gesture label (or 'ALL' to clear all)."""
    db = _get_db()
    deleted = db.delete_samples_for_label(label)
    return {
        "deleted": deleted,
        "label": label.strip().upper(),
        "sample_counts": db.get_sample_counts(),
    }


@app.post("/api/train")
async def train_custom_model_endpoint(req: TrainModelRequest = TrainModelRequest()) -> Dict[str, Any]:
    """Train StaticSignMLP on user-recorded webcam samples and hot-reload the live inference engine."""
    db = _get_db()
    engine = _get_engine()
    try:
        summary = train_from_sqlite_dataset(db=db, epochs=req.epochs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    engine.reload_weights()
    return {
        "status": "trained",
        "is_custom_trained": engine.is_custom_trained,
        "active_classes": engine.classes,
        "metrics": summary,
    }


@app.post("/api/train/reset-demo")
async def reset_demo_model_endpoint() -> Dict[str, Any]:
    """Restore the default 40-class demo PyTorch model and hot-reload the inference engine."""
    engine = _get_engine()
    summary = train_and_save_demo_model(epochs=30, samples_per_class=60, verbose=False)
    engine.reload_weights()
    return {
        "status": "reset_to_demo",
        "is_custom_trained": engine.is_custom_trained,
        "active_classes": engine.classes,
        "metrics": summary,
    }


@app.post("/api/refine", response_model=GrammarRefineResponse)
async def refine_sentence_endpoint(req: GrammarRefineRequest) -> GrammarRefineResponse:
    """Reconstruct raw sign glosses or finger-spelled characters into natural grammatical sentences."""
    refined, corrections = refine_gloss_sentence(req.raw_text, language=req.language)
    return GrammarRefineResponse(
        raw_text=req.raw_text,
        refined_text=refined,
        corrections_applied=corrections,
    )


def process_single_frame_payload(
    payload: FramePayload,
    engine: SignInferenceEngine,
    aggregator: SignStateAggregator,
    db: SignStreamDatabase,
    session_id: str,
) -> InferenceResponse:
    """Execute full normalization -> PyTorch inference -> debounce state machine for one frame."""
    t_start = time.perf_counter()

    # Handle explicit UI control command if provided
    if payload.command:
        cmd_event = aggregator.apply_explicit_command(payload.command)
        refined, _ = refine_gloss_sentence(aggregator.active_text)
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        return InferenceResponse(
            frame_id=payload.frame_id,
            client_timestamp_ms=payload.timestamp_ms,
            inference_latency_ms=latency_ms,
            status=SystemStatus.IDLE_WAITING if not aggregator.active_text else SystemStatus.TRACKING,
            exception_code=ExceptionCode.NONE,
            emitted_event=cmd_event,
            window_fill=len(aggregator.rolling_window),
            window_size=aggregator.window_size,
            agreement_count=0,
            lock_threshold=aggregator.lock_in_threshold,
            cooldown_remaining=max(
                0, aggregator.idle_cooldown_frames - aggregator.idle_frames_since_emission
            ),
            active_text=aggregator.active_text,
            refined_sentence=refined,
            submitted_sentences=list(aggregator.submitted_sentences),
        )

    # Coordinate Pre-processor & Edge-Case Inspection
    norm_res = preprocess_frame_hands(payload.hands)

    raw_pred: Optional[str] = None
    conf: float = 0.0
    top_preds = []
    dynamic_token: Optional[str] = None

    if norm_res.exception_code != ExceptionCode.NO_HAND_DETECTED and not norm_res.suppress_classification:
        raw_pred, conf, top_preds = engine.predict_static(
            norm_res.single_hand_vector,
            top_k=3,
            hint_label=payload.hint_label,
        )
        dynamic_token = engine.step_dynamic(norm_res.two_hand_vector)

    # Debounce & State Aggregator
    agg_out = aggregator.step(
        predicted_label=raw_pred,
        confidence=conf,
        timestamp_ms=payload.timestamp_ms,
        exception_code=norm_res.exception_code,
        exception_message=norm_res.exception_message,
        suppress_classification=norm_res.suppress_classification,
        wrist_x=norm_res.wrist_xy[0],
    )

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    if agg_out.emitted_event:
        try:
            db.log_session_emission(
                session_id=session_id,
                emitted_token=agg_out.emitted_event,
                confidence=agg_out.effective_confidence,
                active_text=agg_out.active_text,
                latency_ms=latency_ms,
            )
        except Exception:
            pass

    return InferenceResponse(
        frame_id=payload.frame_id,
        client_timestamp_ms=payload.timestamp_ms,
        inference_latency_ms=latency_ms,
        status=agg_out.status,
        exception_code=agg_out.exception_code,
        exception_message=agg_out.exception_message,
        raw_prediction=agg_out.effective_prediction,
        confidence=round(agg_out.effective_confidence, 4),
        top_predictions=top_preds,
        dynamic_token=dynamic_token,
        locked_character=agg_out.locked_character,
        emitted_event=agg_out.emitted_event,
        window_fill=agg_out.window_fill,
        window_size=agg_out.window_size,
        agreement_count=agg_out.agreement_count,
        lock_threshold=agg_out.lock_threshold,
        cooldown_remaining=agg_out.cooldown_remaining,
        open_palm_hold_progress=round(agg_out.open_palm_hold_progress, 3),
        active_text=agg_out.active_text,
        refined_sentence=agg_out.refined_sentence,
        submitted_sentences=agg_out.submitted_sentences,
        normalized_preview=[round(float(x), 4) for x in norm_res.single_hand_vector[:6]],
    )


@app.websocket("/ws/sign")
async def websocket_sign_endpoint(websocket: WebSocket) -> None:
    """Bi-directional WebSocket stream for 30 FPS landmark ingestion and state machine updates."""
    await websocket.accept()
    engine = _get_engine()
    db = _get_db()
    aggregator = SignStateAggregator()
    session_id = str(uuid.uuid4())[:8]

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            payload = FramePayload.model_validate(data)
            response = process_single_frame_payload(
                payload=payload,
                engine=engine,
                aggregator=aggregator,
                db=db,
                session_id=session_id,
            )
            await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass

