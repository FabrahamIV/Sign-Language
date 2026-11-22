import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  Cpu,
  Play,
  Sparkles,
  VideoOff,
  AlertTriangle,
  Database,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
} from "lucide-react";
import {
  HandLandmarks,
  InferenceResponse,
  Point3D,
} from "../types/signstream";
import {
  ALPHABET_CLASSES,
  CONTROL_CLASSES,
  DIGIT_CLASSES,
  HAND_CONNECTIONS,
  getSimulatedLandmarks,
} from "../utils/handPrototypes";

interface CameraFeedProps {
  connected: boolean;
  latestResponse: InferenceResponse | null;
  serverPrototypes: Record<string, Point3D[]>;
  latestHandRef?: React.MutableRefObject<HandLandmarks | null>;
  onFrameLandmarks: (hands: HandLandmarks[], hintLabel?: string | null) => void;
  onFpsUpdate: (fps: number) => void;
  onRecordSample: (label: string, hand: HandLandmarks) => Promise<void>;
}

export const CameraFeed: React.FC<CameraFeedProps> = ({
  connected,
  latestResponse,
  serverPrototypes,
  latestHandRef,
  onFrameLandmarks,
  onFpsUpdate,
  onRecordSample,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [mode, setMode] = useState<"webcam" | "simulator">("webcam");
  const [showTools, setShowTools] = useState<boolean>(false);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [mpStatus, setMpStatus] = useState<string>("Loading MediaPipe...");
  const [mpReady, setMpReady] = useState<boolean>(false);

  const [selectedSign, setSelectedSign] = useState<string>("H");
  const [edgeScenario, setEdgeScenario] = useState<string>("NONE");
  const [autoSpelling, setAutoSpelling] = useState<boolean>(false);
  const [savedToast, setSavedToast] = useState<string | null>(null);

  const lastHandsRef = useRef<HandLandmarks[]>([]);
  const landmarkerRef = useRef<any>(null);
  const lastMpTimestampRef = useRef<number>(0);

  const stopCurrentStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  // Initialize MediaPipe HandLandmarker once (local /wasm v0.10.35 + GPU/CPU fallback)
  useEffect(() => {
    let cancelled = false;

    async function loadMediaPipe() {
      if (landmarkerRef.current) {
        setMpReady(true);
        setMpStatus("MediaPipe Ready");
        return;
      }

      try {
        setMpStatus("Loading MediaPipe...");
        const vision = await import("@mediapipe/tasks-vision");

        let filesetResolver: any;
        try {
          filesetResolver = await vision.FilesetResolver.forVisionTasks(
            "/wasm"
          );
        } catch {
          filesetResolver = await vision.FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
          );
        }

        const modelUrls = [
          "/models/hand_landmarker.task",
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        ];

        let createdLandmarker: any = null;
        for (const modelUrl of modelUrls) {
          for (const delegate of ["GPU", "CPU"] as const) {
            try {
              createdLandmarker = await vision.HandLandmarker.createFromOptions(
                filesetResolver,
                {
                  baseOptions: {
                    modelAssetPath: modelUrl,
                    delegate,
                  },
                  runningMode: "VIDEO",
                  numHands: 2,
                  minHandDetectionConfidence: 0.5,
                  minHandPresenceConfidence: 0.5,
                  minTrackingConfidence: 0.5,
                }
              );
              if (createdLandmarker) {
                if (!cancelled) {
                  setMpStatus(`MediaPipe (${delegate})`);
                }
                break;
              }
            } catch {
              // Try next delegate or model URL
            }
          }
          if (createdLandmarker) break;
        }

        if (!createdLandmarker) {
          throw new Error("Unable to initialize HandLandmarker.");
        }

        if (!cancelled) {
          landmarkerRef.current = createdLandmarker;
          setMpReady(true);
        }
      } catch (err: any) {
        if (!cancelled) {
          setMpStatus(`MediaPipe Error: ${err?.message || "Unknown"}`);
        }
      }
    }

    loadMediaPipe();
    return () => {
      cancelled = true;
    };
  }, []);

  // Start physical webcam
  const startWebcam = useCallback(
    async (deviceIdToUse?: string) => {
      setWebcamError(null);
      stopCurrentStream();

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setWebcamError(
          "Camera API unavailable. Please open via http://localhost:5173 or HTTPS."
        );
        return;
      }

      try {
        const primaryConstraints: MediaStreamConstraints = {
          video: deviceIdToUse
            ? {
                deviceId: { exact: deviceIdToUse },
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
              }
            : {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
              },
          audio: false,
        };

        let mediaStream: MediaStream;
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia(
            primaryConstraints
          );
        } catch {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        }

        streamRef.current = mediaStream;

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = allDevices.filter((d) => d.kind === "videoinput");
        setCameras(videoDevices);

        const activeTrack = mediaStream.getVideoTracks()[0];
        const activeDeviceId = activeTrack?.getSettings()?.deviceId;
        if (activeDeviceId && !deviceIdToUse) {
          setSelectedDeviceId(activeDeviceId);
        }

        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          await videoRef.current.play();
          setCameraActive(true);
        }
      } catch (err: any) {
        const errName = err?.name || "";
        let friendlyMsg = err?.message || "Unable to access device camera.";
        if (
          errName === "NotAllowedError" ||
          errName === "PermissionDeniedError"
        ) {
          friendlyMsg =
            "Camera permission denied. Allow camera access in your browser address bar, then click Retry.";
        } else if (
          errName === "NotFoundError" ||
          errName === "DevicesNotFoundError"
        ) {
          friendlyMsg =
            "No camera detected on this device. Connect a webcam or switch to 3D Simulator.";
        } else if (
          errName === "NotReadableError" ||
          errName === "TrackStartError"
        ) {
          friendlyMsg =
            "Camera is in use by another app (e.g. Zoom/OBS). Close it and click Retry.";
        }
        setWebcamError(friendlyMsg);
        setCameraActive(false);
      }
    },
    [stopCurrentStream]
  );

  useEffect(() => {
    if (mode === "webcam") {
      startWebcam(selectedDeviceId || undefined);
    } else {
      stopCurrentStream();
      setShowTools(true);
    }
    return () => {
      stopCurrentStream();
    };
  }, [mode, selectedDeviceId, startWebcam, stopCurrentStream]);

  // Auto-spell demo sequence
  useEffect(() => {
    if (!autoSpelling || mode !== "simulator") return;
    const sequence = [
      "H", "IDLE", "A", "IDLE", "L", "IDLE", "O", "IDLE",
      "SPACE", "IDLE",
      "S", "IDLE", "A", "IDLE", "Y", "IDLE", "A", "IDLE",
    ];
    let stepIdx = 0;
    setSelectedSign(sequence[0]);

    const timer = window.setInterval(() => {
      stepIdx++;
      if (stepIdx >= sequence.length) {
        setAutoSpelling(false);
        setSelectedSign("IDLE");
        window.clearInterval(timer);
        return;
      }
      setSelectedSign(sequence[stepIdx]);
    }, 620);

    return () => window.clearInterval(timer);
  }, [autoSpelling, mode]);

  // 30 FPS Capture, Skeletal Overlay & WebSocket Loop
  useEffect(() => {
    let animId: number;
    let lastSendTime = 0;
    let frameCount = 0;
    let fpsStart = performance.now();

    const renderLoop = (now: number) => {
      animId = requestAnimationFrame(renderLoop);

      if (now - lastSendTime < 32) return;
      lastSendTime = now;

      frameCount++;
      if (now - fpsStart >= 1000) {
        onFpsUpdate(Math.round((frameCount * 1000) / (now - fpsStart)));
        frameCount = 0;
        fpsStart = now;
      }

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      let handsToSend: HandLandmarks[] = [];

      if (mode === "webcam") {
        const videoEl = videoRef.current;
        if (
          videoEl &&
          videoEl.readyState >= 2 &&
          videoEl.videoWidth > 0 &&
          mpReady &&
          landmarkerRef.current
        ) {
          try {
            const mpTs = Math.max(
              Math.round(now),
              lastMpTimestampRef.current + 1
            );
            lastMpTimestampRef.current = mpTs;

            const res = landmarkerRef.current.detectForVideo(videoEl, mpTs);
            if (res && res.landmarks && res.landmarks.length > 0) {
              handsToSend = res.landmarks.map((lms: any[], idx: number) => {
                const handedness =
                  res.handednesses?.[idx]?.[0]?.categoryName || "Right";
                const score = res.handednesses?.[idx]?.[0]?.score ?? 0.95;
                const pts: Point3D[] = lms.map((lm: any) => ({
                  x: Number((1.0 - lm.x).toFixed(5)),
                  y: Number(lm.y.toFixed(5)),
                  z: Number(lm.z.toFixed(5)),
                  visibility:
                    edgeScenario === "LOW_LIGHTING_BLUR" ? 0.35 : 0.98,
                }));
                return { handedness, score, landmarks: pts };
              });
            }
          } catch {
            // Ignore transient frame timing error
          }
        }
      } else {
        // Studio dark grid for 3D Simulator
        const grad = ctx.createRadialGradient(
          width * 0.5,
          height * 0.55,
          30,
          width * 0.5,
          height * 0.5,
          width * 0.72
        );
        grad.addColorStop(0, "#0f172a");
        grad.addColorStop(1, "#050814");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = "rgba(56, 189, 248, 0.06)";
        ctx.lineWidth = 1;
        for (let gx = 0; gx < width; gx += 40) {
          ctx.beginPath();
          ctx.moveTo(gx, 0);
          ctx.lineTo(gx, height);
          ctx.stroke();
        }
        for (let gy = 0; gy < height; gy += 40) {
          ctx.beginPath();
          ctx.moveTo(0, gy);
          ctx.lineTo(width, gy);
          ctx.stroke();
        }

        if (edgeScenario !== "NO_HAND_DETECTED") {
          const pts = getSimulatedLandmarks(
            selectedSign,
            serverPrototypes,
            edgeScenario
          );
          handsToSend = [
            {
              handedness: "Right",
              score: edgeScenario === "LOW_LIGHTING_BLUR" ? 0.38 : 0.99,
              landmarks: pts,
            },
          ];
        }
      }

      // Subtle viewfinder corner brackets for safe bounding zone
      const pad = 22;
      const bLen = 22;
      ctx.save();
      ctx.strokeStyle =
        edgeScenario === "OUT_OF_BOUNDS"
          ? "rgba(244, 63, 94, 0.9)"
          : "rgba(56, 189, 248, 0.35)";
      ctx.lineWidth = 2;
      const corners: [number, number, number, number][] = [
        [pad, pad, 1, 1],
        [width - pad, pad, -1, 1],
        [pad, height - pad, 1, -1],
        [width - pad, height - pad, -1, -1],
      ];
      corners.forEach(([x, y, dx, dy]) => {
        ctx.beginPath();
        ctx.moveTo(x + dx * bLen, y);
        ctx.lineTo(x, y);
        ctx.lineTo(x, y + dy * bLen);
        ctx.stroke();
      });
      ctx.restore();

      // Draw 21-point 3D Hand Skeleton
      handsToSend.forEach((hand) => {
        const lms = hand.landmarks;
        ctx.save();
        ctx.shadowColor = "rgba(34, 211, 238, 0.45)";
        ctx.shadowBlur = 8;
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
        HAND_CONNECTIONS.forEach(([i, j]) => {
          const p1 = lms[i];
          const p2 = lms[j];
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        });
        ctx.restore();

        lms.forEach((pt, idx) => {
          const cx = pt.x * width;
          const cy = pt.y * height;
          const isWrist = idx === 0;
          const isFingertip = [4, 8, 12, 16, 20].includes(idx);

          ctx.beginPath();
          ctx.arc(cx, cy, isWrist ? 6.5 : isFingertip ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fillStyle = isWrist
            ? "#f59e0b"
            : isFingertip
            ? "#38bdf8"
            : "#f8fafc";
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = "#060913";
          ctx.stroke();
        });
      });

      lastHandsRef.current = handsToSend;
      if (latestHandRef) {
        latestHandRef.current = handsToSend[0] || null;
      }
      if (connected) {
        onFrameLandmarks(
          handsToSend,
          mode === "simulator" ? selectedSign : null
        );
      }
    };

    animId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animId);
  }, [
    mode,
    selectedSign,
    edgeScenario,
    connected,
    mpReady,
    serverPrototypes,
    onFrameLandmarks,
    onFpsUpdate,
  ]);

  const handleSaveCurrentSample = async () => {
    const hand = lastHandsRef.current[0];
    if (!hand) return;
    await onRecordSample(selectedSign, hand);
    setSavedToast(`Saved normalized 63-d sample for '${selectedSign}'`);
    window.setTimeout(() => setSavedToast(null), 2500);
  };

  const confidencePct = latestResponse
    ? Math.round(latestResponse.confidence * 100)
    : 0;

  return (
    <div className="studio-card p-4 sm:p-5 flex flex-col gap-4">
      {/* Top Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Segmented Mode Switcher */}
        <div className="inline-flex items-center p-1 rounded-xl bg-slate-950/90 border border-slate-800">
          <button
            type="button"
            onClick={() => setMode("webcam")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              mode === "webcam"
                ? "bg-gradient-to-r from-cyan-500 to-sky-500 text-slate-950 shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            Live Webcam
          </button>
          <button
            type="button"
            onClick={() => setMode("simulator")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              mode === "simulator"
                ? "bg-gradient-to-r from-cyan-500 to-sky-500 text-slate-950 shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            3D Simulator
          </button>
        </div>

        {/* Camera Controls / Status */}
        <div className="flex flex-wrap items-center gap-2">
          {mode === "webcam" && cameras.length > 0 && (
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              className="bg-slate-950 text-slate-200 text-xs border border-slate-800 rounded-lg px-2.5 py-1.5 max-w-[190px] truncate focus:outline-none focus:border-cyan-500"
              title="Select Camera"
            >
              {cameras.map((cam, idx) => (
                <option key={cam.deviceId || idx} value={cam.deviceId}>
                  {cam.label || `Camera ${idx + 1}`}
                </option>
              ))}
            </select>
          )}

          {mode === "webcam" && (
            <button
              type="button"
              onClick={() => startWebcam(selectedDeviceId || undefined)}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700/80 transition"
              title="Reconnect Camera"
            >
              <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
              <span className="hidden sm:inline">Retry</span>
            </button>
          )}

          <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium bg-slate-950/80 text-slate-300 border border-slate-800">
            <span
              className={`w-2 h-2 rounded-full ${
                mode === "webcam"
                  ? cameraActive
                    ? "bg-emerald-400 animate-pulse"
                    : "bg-amber-400"
                  : "bg-cyan-400 animate-pulse"
              }`}
            />
            {mode === "webcam"
              ? cameraActive
                ? mpStatus
                : "Waiting for Camera"
              : "Simulator Active"}
          </span>
        </div>
      </div>

      {/* Camera Error Banner */}
      {webcamError && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs bg-amber-500/10 border border-amber-500/30 text-amber-200 px-3.5 py-2.5 rounded-xl">
          <div className="flex items-center gap-2">
            <VideoOff className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{webcamError}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => startWebcam(selectedDeviceId || undefined)}
              className="px-3 py-1 rounded-lg bg-amber-500 text-slate-950 font-semibold hover:bg-amber-400 transition"
            >
              Retry
            </button>
            <button
              type="button"
              onClick={() => setMode("simulator")}
              className="px-3 py-1 rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700 transition"
            >
              Use Simulator
            </button>
          </div>
        </div>
      )}

      {/* Studio Camera Viewport */}
      <div className="relative w-full aspect-[4/3] bg-[#050814] rounded-xl overflow-hidden border border-slate-800/90 shadow-inner">
        <video
          ref={videoRef}
          className={`absolute inset-0 w-full h-full object-cover -scale-x-100 ${
            mode === "webcam" ? "block" : "hidden"
          }`}
          playsInline
          autoPlay
          muted
          width={640}
          height={480}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />

        {/* Floating HUD Top-Left: Live Sign Prediction */}
        <div className="absolute top-3.5 left-3.5 flex items-center gap-3 bg-slate-950/75 backdrop-blur-md border border-white/10 px-3.5 py-2 rounded-xl shadow-lg">
          <div className="w-11 h-11 rounded-lg bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center text-2xl font-extrabold font-mono text-cyan-300">
            {latestResponse?.raw_prediction || "—"}
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Detected Sign
            </span>
            <div className="flex items-center gap-2 mt-0.5">
              <span
                className={`text-xs font-mono font-bold ${
                  confidencePct >= 85 ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {latestResponse ? `${confidencePct}%` : "0%"}
              </span>
              <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-100 ${
                    confidencePct >= 85 ? "bg-emerald-400" : "bg-amber-400"
                  }`}
                  style={{ width: `${confidencePct}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Floating HUD Top-Right: State Machine Pill */}
        <div className="absolute top-3.5 right-3.5 flex flex-col items-end gap-1.5">
          <span
            className={`px-3 py-1 rounded-full text-[11px] font-semibold tracking-wide border backdrop-blur-md shadow ${
              latestResponse?.status === "LOCKED_IN"
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-400/40"
                : latestResponse?.status === "COOLDOWN"
                ? "bg-amber-500/20 text-amber-300 border-amber-400/40"
                : latestResponse?.status === "WARNING"
                ? "bg-rose-500/20 text-rose-300 border-rose-400/40"
                : "bg-slate-950/75 text-slate-300 border-white/10"
            }`}
          >
            {latestResponse?.status?.replace("_", " ") || "IDLE WAITING"}
          </span>

          {latestResponse?.cooldown_remaining ? (
            <span className="text-[10px] font-mono bg-slate-950/80 text-amber-300 px-2.5 py-0.5 rounded-full border border-amber-500/30 backdrop-blur-md">
              Cooldown: {latestResponse.cooldown_remaining}/10 IDLE
            </span>
          ) : null}
        </div>

        {/* Floating HUD Bottom: Edge-Case Warning */}
        {latestResponse?.exception_code &&
          latestResponse.exception_code !== "NONE" && (
            <div className="absolute bottom-3.5 inset-x-3.5 flex items-center gap-2 bg-rose-950/85 border border-rose-500/40 text-rose-100 px-3.5 py-2 rounded-xl text-xs backdrop-blur-md shadow-lg">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span className="font-mono font-semibold text-rose-300">
                {latestResponse.exception_code}
              </span>
              <span className="text-rose-100/90 truncate">
                {latestResponse.exception_message}
              </span>
            </div>
          )}
      </div>

      {/* Collapsible Simulator / Dataset Tools Drawer */}
      <div className="border-t border-slate-800/80 pt-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setShowTools((prev) => !prev)}
            className="inline-flex items-center gap-2 text-xs font-semibold text-slate-300 hover:text-white transition"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
            <span>
              {mode === "simulator"
                ? "Sign Pose Simulator & Edge-Case Controls"
                : "Dataset Recorder & Sign Simulator Tools"}
            </span>
            {showTools ? (
              <ChevronUp className="w-4 h-4 text-slate-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-500" />
            )}
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setMode("simulator");
                setShowTools(true);
                setEdgeScenario("NONE");
                setAutoSpelling((prev) => !prev);
              }}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                autoSpelling
                  ? "bg-amber-500 text-slate-950"
                  : "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/20"
              }`}
            >
              <Play className="w-3 h-3" />
              {autoSpelling ? "Stop Demo" : "Demo 'HALO SAYA'"}
            </button>

            <button
              type="button"
              onClick={handleSaveCurrentSample}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
              title="Save current normalized 63-d hand landmark vector to SQLite"
            >
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              Save Sample ({selectedSign})
            </button>
          </div>
        </div>

        {savedToast && (
          <div className="mt-2.5 text-xs text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 px-3 py-2 rounded-lg flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            {savedToast}
          </div>
        )}

        {showTools && (
          <div className="mt-3 flex flex-col gap-3 bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5">
            {/* Alphabet Grid: 7 cols on mobile, 13 cols (2 clean rows) on sm+ */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Alphabet (A–Z)
              </span>
              <div className="grid grid-cols-7 sm:grid-cols-13 gap-1.5">
                {ALPHABET_CLASSES.map((letter) => (
                  <button
                    key={letter}
                    type="button"
                    onClick={() => {
                      setAutoSpelling(false);
                      setEdgeScenario("NONE");
                      setSelectedSign(letter);
                    }}
                    className={`h-8 rounded-lg text-xs font-mono font-bold transition border ${
                      selectedSign === letter && edgeScenario === "NONE"
                        ? "bg-cyan-500 text-slate-950 border-cyan-400 shadow-sm shadow-cyan-500/30"
                        : "bg-slate-900/90 text-slate-300 border-slate-800 hover:border-slate-600 hover:text-white"
                    }`}
                  >
                    {letter}
                  </button>
                ))}
              </div>
            </div>

            {/* Digits & Control Gestures */}
            <div className="flex flex-wrap items-center gap-1.5">
              {DIGIT_CLASSES.map((digit) => (
                <button
                  key={digit}
                  type="button"
                  onClick={() => {
                    setAutoSpelling(false);
                    setEdgeScenario("NONE");
                    setSelectedSign(digit);
                  }}
                  className={`w-8 h-7 rounded-lg text-xs font-mono font-semibold transition border ${
                    selectedSign === digit && edgeScenario === "NONE"
                      ? "bg-indigo-500 text-white border-indigo-400"
                      : "bg-slate-900/90 text-slate-400 border-slate-800 hover:border-slate-600 hover:text-slate-200"
                  }`}
                >
                  {digit}
                </button>
              ))}

              <div className="h-5 w-px bg-slate-800 mx-1 hidden sm:block" />

              {CONTROL_CLASSES.map((ctrl) => (
                <button
                  key={ctrl}
                  type="button"
                  onClick={() => {
                    setAutoSpelling(false);
                    setEdgeScenario("NONE");
                    setSelectedSign(ctrl);
                  }}
                  className={`px-2.5 h-7 rounded-lg text-[11px] font-mono font-semibold transition border ${
                    selectedSign === ctrl && edgeScenario === "NONE"
                      ? "bg-emerald-500 text-slate-950 border-emerald-400"
                      : "bg-slate-900/90 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/15"
                  }`}
                >
                  {ctrl}
                </button>
              ))}
            </div>

            {/* Edge-Case Triggers */}
            <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-800/80">
              <span className="text-[11px] text-slate-500 mr-1">
                Edge-Case Tests:
              </span>
              {[
                { code: "NONE", label: "Normal" },
                { code: "NO_HAND_DETECTED", label: "No Hand" },
                { code: "LOW_LIGHTING_BLUR", label: "Low Light" },
                { code: "OUT_OF_BOUNDS", label: "Out of Bounds" },
                { code: "SELF_OCCLUSION", label: "Self-Occlusion" },
              ].map((item) => (
                <button
                  key={item.code}
                  type="button"
                  onClick={() => {
                    setAutoSpelling(false);
                    setEdgeScenario(item.code);
                  }}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition border ${
                    edgeScenario === item.code
                      ? "bg-rose-500/20 text-rose-200 border-rose-500/40"
                      : "bg-slate-900/80 text-slate-400 border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
