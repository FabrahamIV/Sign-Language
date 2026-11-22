# SignStream — Real-Time Sign Language Translation System

**Version:** 1.0  
**Document Type:** Project Design & Architecture Document (PDD) & Implementation Repository  
**Status:** MVP Implemented & Ready for Execution

---

## 1. Overview & Primary Objective

**SignStream** is a low-latency, camera-based computer vision application that detects 3D hand landmarks in real time, classifies static finger-spelling alphabets (SIBI / ASL `A–Z`, `0–9`) and dynamic/control gestures, and translates them into structured, readable text with Text-to-Speech (TTS) output.

### Key Capabilities
- **Real-Time 30 FPS Ingestion**: Webcam capture with HTML5 `<canvas>` skeletal overlay and MediaPipe Hands 21-keypoint 3D extraction (plus built-in interactive gesture simulator for headless/webcam-free testing).
- **Mathematical Coordinate Normalization**: Translation invariance (wrist origin alignment $\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0$) and scale invariance ($L_2$ max-distance normalization $\mathbf{p}^{\text{norm}}_i = \frac{\mathbf{p}'_i}{\max_j \|\mathbf{p}'_j\|_2 + \epsilon}$).
- **PyTorch Inference Engine**:
  - `StaticSignMLP`: 3-layer MLP with BatchNorm and Dropout operating on 63-dimensional normalized single-hand vectors.
  - `DynamicSignLSTM`: Temporal LSTM classifier operating on $(30, 126)$ two-hand rolling windows.
- **Deterministic Debounce & State Machine**:
  - $N = 15$ rolling frame window queue.
  - $0.85$ softmax confidence gate.
  - $\ge 12 / 15$ frame lock-in threshold.
  - $10$-frame `IDLE` cooldown to prevent duplicate emissions.
  - Control gestures: `OPEN_PALM_HOLD` (1.5s Space), `FIST_SWIPE_LEFT` (Backspace), `THUMBS_UP_HOLD` (Submit/Clear).
- **Edge-Case & Exception Handling**:
  - `NO_HAND_DETECTED`, `LOW_LIGHTING_BLUR` (visibility $< 0.50$), `OUT_OF_BOUNDS` ($< 0.02$ or $> 0.98$), `TRANSITION_NOISE` ($> 30$ frames ambiguous), and `SELF_OCCLUSION` (Z-depth spike / finger overlap with 5-frame temporal inertia).

---

## 2. System Architecture

```
[Device Camera] 
       │ (30 FPS RGB Stream)
       ▼
[Frontend Client (React + Vite + Tailwind)] ──► Renders skeletal overlay & UI
       │ (Client-Side MediaPipe Hands -> WebSocket JSON Stream)
       ▼
[Coordinate Pre-processor (normalizer.py)]
       │ (Wrist-centering, L2 scale normalization, edge-case validation)
       ▼
[PyTorch Inference Engine (model.py)]
       ├──► Static Classifier (3-Layer MLP) ──► Single Token (A-Z, 0-9, Control)
       └──► Dynamic Classifier (LSTM)       ──► Temporal Gesture / Swipe Detection
       │
       ▼
[Debounce & State Aggregator (aggregator.py)]
       │ (Confidence >= 0.85, >= 12/15 frames lock-in, 10-frame IDLE cooldown)
       ▼
[Text Output + Grammar Refiner + Web Speech TTS]
```

---

## 3. Project Directory Structure

```text
Sign_Language/
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml              # Managed via uv / standard PEP 621
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point, REST & WebSocket routes
│   │   ├── database.py             # SQLite persistence for sessions & model metrics
│   │   ├── schemas.py              # Pydantic models for landmark payloads & state
│   │   ├── vision/
│   │   │   ├── extractor.py        # Server-side MediaPipe wrapper
│   │   │   ├── normalizer.py       # Wrist-centering & L2 scaling math + edge cases
│   │   │   └── aggregator.py       # Debounce, rolling buffer & state machine
│   │   └── ml/
│   │       ├── model.py            # PyTorch MLP & LSTM architectures
│   │       ├── train.py            # Training loop, evaluation & checkpoint export
│   │       └── weights/            # Saved .pt / .onnx model checkpoints
│   ├── scripts/
│   │   ├── collect_dataset.py      # OpenCV tool to record & label custom signs
│   │   └── generate_demo_data.py   # Anatomically grounded synthetic dataset generator
│   └── tests/
│       └── test_pipeline.py        # Unit tests for math, state machine & inference
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── components/
│       │   ├── CameraFeed.tsx      # Webcam stream, MediaPipe & Canvas skeletal overlay
│       │   ├── TranscriptBox.tsx   # Live character lock-in, sentence & TTS controls
│       │   └── MetricsPanel.tsx    # FPS, latency (<35ms KPI), confidence & edge badges
│       └── hooks/
│           └── useSignSocket.ts    # Bi-directional WebSocket streaming hook
└── docker-compose.yml
```

---

## 4. Quickstart Instructions

### Option A: Local Development

#### 1. Backend Setup (FastAPI + PyTorch)
```bash
cd backend
# Using uv (or python -m venv .venv)
uv sync
# Generate initial dataset and pre-train the MLP checkpoint (>97% validation F1)
uv run python -m app.ml.train --generate-demo
# Start the FastAPI WebSocket server on port 8000
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup (React + Vite + Tailwind CSS)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

#### 3. Collect Custom Dataset with OpenCV (Optional)
```bash
cd backend
uv run python scripts/collect_dataset.py --output data/landmarks_dataset.csv --samples 50
```

### Option B: Docker Compose
```bash
docker compose up --build
```
- **Frontend UI**: `http://localhost:5173`
- **Backend API & Docs**: `http://localhost:8000/docs`
- **WebSocket Stream**: `ws://localhost:8000/ws/sign`

