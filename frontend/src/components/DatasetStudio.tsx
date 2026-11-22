import React, { useRef, useState } from "react";
import {
  Brain,
  CheckCircle2,
  Database,
  Loader2,
  RotateCcw,
  Sparkles,
  Trash2,
  Video,
  AlertCircle,
} from "lucide-react";
import { HandLandmarks } from "../types/signstream";

interface DatasetStudioProps {
  latestHandRef: React.MutableRefObject<HandLandmarks | null>;
  sampleCounts: Record<string, number>;
  isCustomTrained: boolean;
  activeClasses: string[];
  onRecordBatch: (label: string, hands: HandLandmarks[]) => Promise<any>;
  onDeleteSamples: (label: string) => Promise<any>;
  onTrainModel: (epochs?: number) => Promise<any>;
  onResetDemo: () => Promise<any>;
}

const QUICK_LABELS = [
  "HALO",
  "SAYA",
  "TERIMA_KASIH",
  "TOLONG",
  "MAAF",
  "IDLE",
  "SPACE",
  "BACKSPACE",
  "SUBMIT",
  "A",
  "B",
  "C",
  "D",
  "E",
];

const TARGET_BURST_FRAMES = 40;

export const DatasetStudio: React.FC<DatasetStudioProps> = ({
  latestHandRef,
  sampleCounts,
  isCustomTrained,
  activeClasses,
  onRecordBatch,
  onDeleteSamples,
  onTrainModel,
  onResetDemo,
}) => {
  const [labelInput, setLabelInput] = useState<string>("HALO");
  const [recordingPhase, setRecordingPhase] = useState<
    "idle" | "countdown" | "recording" | "saving"
  >("idle");
  const [countdown, setCountdown] = useState<number>(0);
  const [capturedCount, setCapturedCount] = useState<number>(0);
  const [training, setTraining] = useState<boolean>(false);
  const [resetting, setResetting] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const cancelRef = useRef<boolean>(false);

  const recordedEntries = Object.entries(sampleCounts);
  const totalSamples = recordedEntries.reduce((acc, [, n]) => acc + n, 0);
  const eligibleClassCount = recordedEntries.filter(([, n]) => n >= 5).length;

  const startBurstRecording = async () => {
    const cleanLabel = labelInput.trim().toUpperCase().replace(/\s+/g, "_");
    if (!cleanLabel) {
      setStatusMsg({
        type: "error",
        text: "Enter a gesture label first (e.g. HALO, A, B, IDLE).",
      });
      return;
    }

    cancelRef.current = false;
    setStatusMsg(null);
    setCapturedCount(0);

    // 2-second countdown so the user can position their hand in front of the camera
    setRecordingPhase("countdown");
    for (let sec = 2; sec >= 1; sec--) {
      if (cancelRef.current) {
        setRecordingPhase("idle");
        return;
      }
      setCountdown(sec);
      await new Promise((r) => setTimeout(r, 750));
    }

    setRecordingPhase("recording");
    const collected: HandLandmarks[] = [];
    let missedTicks = 0;

    while (collected.length < TARGET_BURST_FRAMES && !cancelRef.current) {
      const currentHand = latestHandRef.current;
      if (currentHand && currentHand.landmarks.length === 21) {
        collected.push(currentHand);
        setCapturedCount(collected.length);
        missedTicks = 0;
      } else {
        missedTicks++;
        if (missedTicks > 120) {
          setRecordingPhase("idle");
          setStatusMsg({
            type: "error",
            text: "No hand detected by camera during recording. Make sure your hand is visible in the frame and try again.",
          });
          return;
        }
      }
      await new Promise((r) => setTimeout(r, 40));
    }

    if (cancelRef.current || collected.length === 0) {
      setRecordingPhase("idle");
      return;
    }

    setRecordingPhase("saving");
    try {
      await onRecordBatch(cleanLabel, collected);
      setStatusMsg({
        type: "success",
        text: `Recorded ${collected.length} real webcam frames for "${cleanLabel}". Record at least 2 gestures, then click Train Model!`,
      });
    } catch (err: any) {
      setStatusMsg({
        type: "error",
        text: err?.message || "Failed to save recorded frames.",
      });
    } finally {
      setRecordingPhase("idle");
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setStatusMsg(null);
    try {
      const res = await onTrainModel(45);
      const f1 = ((res.metrics?.val_f1_macro ?? 0) * 100).toFixed(1);
      const acc = ((res.metrics?.val_accuracy ?? 0) * 100).toFixed(1);
      setStatusMsg({
        type: "success",
        text: `Model trained on your ${res.active_classes.length} gestures (${res.active_classes.join(
          ", "
        )})! Val Accuracy: ${acc}%, Macro-F1: ${f1}%. Live camera now uses your custom model!`,
      });
    } catch (err: any) {
      setStatusMsg({
        type: "error",
        text: err?.message || "Training failed.",
      });
    } finally {
      setTraining(false);
    }
  };

  const handleResetDemo = async () => {
    setResetting(true);
    setStatusMsg(null);
    try {
      await onResetDemo();
      setStatusMsg({
        type: "success",
        text: "Restored default 40-class demo model.",
      });
    } catch (err: any) {
      setStatusMsg({
        type: "error",
        text: err?.message || "Reset failed.",
      });
    } finally {
      setResetting(false);
    }
  };

  const burstPct = Math.round((capturedCount / TARGET_BURST_FRAMES) * 100);

  return (
    <div className="studio-card p-4 sm:p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center">
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              Custom Gesture Dataset &amp; Training Studio
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  isCustomTrained
                    ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                    : "bg-slate-800 text-slate-400 border-slate-700"
                }`}
              >
                {isCustomTrained
                  ? `Custom Model Active (${activeClasses.length} signs)`
                  : "Default Demo Weights"}
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">
              Record your own hand gestures from the webcam (letters or full words) and train PyTorch live.
            </p>
          </div>
        </div>
      </div>

      {/* Step 1: Choose or Type Gesture Label + Burst Record */}
      <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3.5 flex flex-col gap-3">
        <div className="flex flex-col sm:flex-row gap-2">
          <div className="flex-1 flex flex-col gap-1">
            <label
              htmlFor="gesture-label-input"
              className="text-[11px] font-semibold uppercase tracking-wider text-slate-400"
            >
              1. Gesture Label (Letter or Custom Word)
            </label>
            <input
              id="gesture-label-input"
              type="text"
              value={labelInput}
              onChange={(e) => setLabelInput(e.target.value.toUpperCase())}
              placeholder="e.g. HALO, SAYA, A, B, IDLE..."
              maxLength={24}
              className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm font-mono font-bold text-white placeholder:text-slate-600 focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div className="flex items-end">
            {recordingPhase === "idle" ? (
              <button
                type="button"
                onClick={startBurstRecording}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 text-slate-950 shadow-lg shadow-cyan-500/20 transition"
              >
                <Video className="w-4 h-4" />
                Record {TARGET_BURST_FRAMES} Frames
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  cancelRef.current = true;
                }}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-rose-500 hover:bg-rose-400 text-white transition"
              >
                <Loader2 className="w-4 h-4 animate-spin" />
                {recordingPhase === "countdown"
                  ? `Hold Pose... ${countdown}`
                  : recordingPhase === "recording"
                  ? `Capturing ${capturedCount}/${TARGET_BURST_FRAMES}`
                  : "Saving..."}
              </button>
            )}
          </div>
        </div>

        {/* Live Recording Progress Bar */}
        {recordingPhase !== "idle" && (
          <div className="flex flex-col gap-1">
            <div className="flex justify-between text-[11px] font-mono text-cyan-300">
              <span>
                {recordingPhase === "countdown"
                  ? `Get ready! Hold "${labelInput}" in front of camera...`
                  : `Capturing real MediaPipe landmarks for "${labelInput}" (slightly move/tilt hand for better accuracy)`}
              </span>
              <span>{burstPct}%</span>
            </div>
            <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-75"
                style={{ width: `${burstPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Quick Preset Chips */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-slate-500 mr-1">Quick Pick:</span>
          {QUICK_LABELS.map((lbl) => (
            <button
              key={lbl}
              type="button"
              onClick={() => setLabelInput(lbl)}
              className={`px-2 py-0.5 rounded-md text-[11px] font-mono font-semibold transition border ${
                labelInput === lbl
                  ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                  : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
              }`}
            >
              {lbl}
            </button>
          ))}
        </div>
      </div>

      {/* Step 2: Recorded Dataset Inventory */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300">
            2. Your Recorded Dataset ({totalSamples} samples across{" "}
            {recordedEntries.length} gestures)
          </span>
          {recordedEntries.length > 0 && (
            <button
              type="button"
              onClick={() => onDeleteSamples("ALL")}
              className="text-[11px] text-rose-400 hover:text-rose-300 inline-flex items-center gap-1 transition"
            >
              <Trash2 className="w-3 h-3" />
              Clear All
            </button>
          )}
        </div>

        {recordedEntries.length === 0 ? (
          <div className="text-xs text-slate-500 bg-slate-950/60 border border-dashed border-slate-800 rounded-xl px-3.5 py-3">
            No custom webcam samples recorded yet. Pick a gesture above (e.g.{" "}
            <span className="font-mono text-cyan-400">HALO</span>,{" "}
            <span className="font-mono text-cyan-400">SAYA</span>, and{" "}
            <span className="font-mono text-cyan-400">IDLE</span>), hold your
            hand in front of the camera, and click{" "}
            <span className="text-slate-200 font-semibold">
              Record {TARGET_BURST_FRAMES} Frames
            </span>
            .
          </div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {recordedEntries.map(([lbl, count]) => (
              <span
                key={lbl}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-950 border border-slate-800 text-slate-200"
              >
                <span className="font-bold text-cyan-300">{lbl}</span>
                <span className="text-[11px] text-emerald-400">{count}</span>
                <button
                  type="button"
                  onClick={() => onDeleteSamples(lbl)}
                  className="text-slate-500 hover:text-rose-400 ml-0.5 transition"
                  title={`Delete samples for ${lbl}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Step 3: Train & Hot-Reload PyTorch Model */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80">
        <button
          type="button"
          onClick={handleTrain}
          disabled={training || eligibleClassCount < 2}
          className={`flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition ${
            eligibleClassCount >= 2
              ? "bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-lg shadow-emerald-500/20"
              : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
          }`}
        >
          {training ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Training PyTorch MLP...
            </>
          ) : (
            <>
              <Brain className="w-4 h-4" />
              3. Train Model on My Gestures ({eligibleClassCount} ready)
            </>
          )}
        </button>

        {isCustomTrained && (
          <button
            type="button"
            onClick={handleResetDemo}
            disabled={resetting}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            {resetting ? "Resetting..." : "Reset Demo Weights"}
          </button>
        )}
      </div>

      {/* Status Feedback Banner */}
      {statusMsg && (
        <div
          className={`flex items-start gap-2 text-xs px-3.5 py-2.5 rounded-xl border ${
            statusMsg.type === "success"
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-200"
              : "bg-rose-500/10 border-rose-500/30 text-rose-200"
          }`}
        >
          {statusMsg.type === "success" ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          )}
          <span>{statusMsg.text}</span>
        </div>
      )}

      {!isCustomTrained && eligibleClassCount >= 2 && (
        <div className="flex items-center gap-1.5 text-[11px] text-cyan-300">
          <Sparkles className="w-3.5 h-3.5 shrink-0" />
          <span>
            Ready! Click <strong>Train Model on My Gestures</strong> to activate
            real-time recognition for your recorded signs.
          </span>
        </div>
      )}
    </div>
  );
};
