"""Unit and integration tests for SignStream math normalization, state machine, PyTorch MLP, and WebSocket API."""

from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.ml.model import STATIC_CLASSES, SignInferenceEngine, build_canonical_hand_prototypes, get_prototype_landmarks_for_label
from app.schemas import ExceptionCode, HandLandmarks, Point3D, SystemStatus
from app.vision.aggregator import SignStateAggregator
from app.vision.normalizer import inspect_hand_exceptions, normalize_hand_array, preprocess_frame_hands


def test_translation_and_scale_invariance() -> None:
    """Verify p'_i = p_i - p_0 and L2 max-distance scaling produce identical 63-d features."""
    protos = build_canonical_hand_prototypes()
    base_hand = protos["A"]  # (21, 3)

    # Shift hand by (+12.5, -8.3, +3.1) and scale by 2.75x
    shifted_and_scaled = (base_hand - base_hand[0:1, :]) * 2.75 + np.array([[12.5, -8.3, 3.1]], dtype=np.float32)

    vec1 = normalize_hand_array(base_hand)
    vec2 = normalize_hand_array(shifted_and_scaled)

    assert vec1.shape == (63,)
    assert vec2.shape == (63,)
    # Wrist (indices 0, 1, 2) must be exactly (0, 0, 0)
    assert np.allclose(vec1[:3], 0.0, atol=1e-7)
    assert np.allclose(vec2[:3], 0.0, atol=1e-7)
    # Max L2 distance from wrist across the 21 landmarks must equal 1.0
    max_l2 = np.max(np.linalg.norm(vec1.reshape(21, 3), axis=1))
    assert np.isclose(max_l2, 1.0, atol=1e-5)
    # Invariance check
    assert np.allclose(vec1, vec2, atol=1e-5)


def test_edge_cases_detection() -> None:
    """Verify PDD Section 7 triggers: NO_HAND_DETECTED, LOW_LIGHTING_BLUR, OUT_OF_BOUNDS, SELF_OCCLUSION."""
    # 1. No Hand Detected
    res_empty = preprocess_frame_hands([])
    assert res_empty.exception_code == ExceptionCode.NO_HAND_DETECTED
    assert res_empty.suppress_classification is True

    # 2. Low Lighting / Blur (visibility < 0.50)
    low_vis_pts = [Point3D(x=0.5, y=0.5, z=0.0, visibility=0.35) for _ in range(21)]
    hand_low_vis = HandLandmarks(handedness="Right", score=0.42, landmarks=low_vis_pts)
    code, msg, suppress = inspect_hand_exceptions(hand_low_vis)
    assert code == ExceptionCode.LOW_LIGHTING_BLUR
    assert "Low visibility" in (msg or "")
    assert suppress is False

    # 3. Out-of-Bounds Hand (coordinate < 0.02 or > 0.98)
    oob_pts = get_prototype_landmarks_for_label("B", jitter_std=0.0)
    oob_pts[8] = Point3D(x=0.995, y=0.5, z=0.0, visibility=0.99)
    hand_oob = HandLandmarks(handedness="Right", score=0.98, landmarks=oob_pts)
    code_oob, _, suppress_oob = inspect_hand_exceptions(hand_oob)
    assert code_oob == ExceptionCode.OUT_OF_BOUNDS
    assert suppress_oob is True

    # 4. Self-Occlusion (Z-depth variance spike)
    occ_pts = get_prototype_landmarks_for_label("C", jitter_std=0.0)
    occ_pts[4] = Point3D(x=0.5, y=0.5, z=1.45, visibility=0.99)
    hand_occ = HandLandmarks(handedness="Right", score=0.98, landmarks=occ_pts)
    code_occ, _, _ = inspect_hand_exceptions(hand_occ)
    assert code_occ == ExceptionCode.SELF_OCCLUSION


def test_state_machine_debounce_and_cooldown() -> None:
    """Verify 15-frame window, 0.85 confidence gate, >=12 lock-in, and 10-frame IDLE cooldown."""
    agg = SignStateAggregator(
        window_size=15,
        confidence_threshold=0.85,
        lock_in_threshold=12,
        idle_cooldown_frames=10,
    )

    t = 1000.0
    # 1. Rapid hand transitions should NOT emit false positives
    for ch in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        out = agg.step(ch, confidence=0.92, timestamp_ms=t)
        t += 33.3
        assert out.emitted_event is None
    assert agg.active_text == ""

    # 2. Hold 'H' for 12 frames at confidence 0.95 -> Locks in 'H'
    agg.reset_buffer()
    emitted = []
    for _ in range(12):
        out = agg.step("H", confidence=0.95, timestamp_ms=t)
        t += 33.3
        if out.emitted_event:
            emitted.append(out.emitted_event)
    assert emitted == ["H"]
    assert agg.active_text == "H"

    # 3. Continuing to hold 'H' for 20 more frames without IDLE must be blocked by cooldown
    for _ in range(20):
        out = agg.step("H", confidence=0.96, timestamp_ms=t)
        t += 33.3
        assert out.emitted_event is None
    assert agg.active_text == "H"

    # 4. Return to IDLE for 10 frames -> resets cooldown so 'H' can be emitted again
    for _ in range(10):
        out = agg.step("IDLE", confidence=0.95, timestamp_ms=t)
        t += 33.3
    assert out.cooldown_remaining == 0

    for _ in range(12):
        out = agg.step("H", confidence=0.95, timestamp_ms=t)
        t += 33.3
    assert agg.active_text == "HH"

    # 5. Control gesture: OPEN_PALM_HOLD for 1.5s inserts a space
    out = agg.step("OPEN_PALM_HOLD", confidence=0.94, timestamp_ms=10_000.0)
    assert out.emitted_event is None
    out = agg.step("OPEN_PALM_HOLD", confidence=0.94, timestamp_ms=11_550.0)
    assert out.emitted_event == "SPACE"
    assert agg.active_text == "HH "

    # 6. Control gesture: BACKSPACE deletes last character
    agg.reset_buffer()
    for _ in range(12):
        out = agg.step("BACKSPACE", confidence=0.95, timestamp_ms=t)
        t += 33.3
    assert agg.active_text == "HH"

    # 7. Control gesture: SUBMIT moves active_text into submitted_sentences
    agg.reset_buffer()
    for _ in range(12):
        out = agg.step("SUBMIT", confidence=0.95, timestamp_ms=t)
        t += 33.3
    assert agg.active_text == ""
    assert len(agg.submitted_sentences) == 1


def test_websocket_streaming_and_kpi_latency() -> None:
    """Verify end-to-end WebSocket inference accuracy and <35ms latency KPI."""
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["num_static_classes"] == len(STATIC_CLASSES)

    with client.websocket_connect("/ws/sign") as ws:
        pts_a = get_prototype_landmarks_for_label("A", jitter_std=0.001)
        emitted_char = None
        for frame_id in range(1, 13):
            payload = {
                "frame_id": frame_id,
                "timestamp_ms": frame_id * 33.3,
                "hands": [
                    {
                        "handedness": "Right",
                        "score": 0.99,
                        "landmarks": [p.model_dump() for p in pts_a],
                    }
                ],
            }
            ws.send_json(payload)
            resp = ws.receive_json()
            assert resp["inference_latency_ms"] < 35.0
            assert resp["raw_prediction"] == "A"
            assert resp["confidence"] >= 0.85
            if resp["emitted_event"]:
                emitted_char = resp["emitted_event"]

        assert emitted_char == "A"
        assert resp["active_text"] == "A"

