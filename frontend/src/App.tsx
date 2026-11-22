import React, { useRef, useState } from "react";
import { Hand, Wifi, WifiOff, Zap, Gauge } from "lucide-react";
import { CameraFeed } from "./components/CameraFeed";
import { DatasetStudio } from "./components/DatasetStudio";
import { TranscriptBox } from "./components/TranscriptBox";
import { MetricsPanel } from "./components/MetricsPanel";
import { useSignSocket } from "./hooks/useSignSocket";
import { HandLandmarks } from "./types/signstream";

export const App: React.FC = () => {
  const {
    connected,
    latestResponse,
    rttMs,
    modelMetrics,
    serverPrototypes,
    sampleCounts,
    isCustomTrained,
    activeClasses,
    sendFrameLandmarks,
    sendCommand,
    recordSampleToDb,
    recordSamplesBatch,
    deleteSamples,
    trainCustomModel,
    resetToDemoModel,
  } = useSignSocket();

  const [fps, setFps] = useState<number>(30);
  const latestHandRef = useRef<HandLandmarks | null>(null);
  const inferenceMs = latestResponse?.inference_latency_ms ?? 0;

  return (
    <div className="min-h-screen text-slate-100 flex flex-col">
      {/* Sleek Studio Navbar */}
      <header className="border-b border-slate-800/70 bg-[#060913]/80 backdrop-blur-xl sticky top-0 z-30">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 via-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/25 ring-1 ring-white/15">
              <Hand className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-white leading-none">
                  SignStream
                </h1>
                <span className="text-[10px] font-mono font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  Real-Time AI
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5 hidden sm:block">
                SIBI / BISINDO / ASL Sign Language to Text Translator
              </p>
            </div>
          </div>

          {/* Live Status Pills */}
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="hidden md:flex items-center gap-3 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-mono text-slate-300">
              <span className="inline-flex items-center gap-1.5">
                <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                {fps} FPS
              </span>
              <span className="w-px h-3.5 bg-slate-700" />
              <span className="inline-flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                {inferenceMs.toFixed(1)} ms
              </span>
            </div>

            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border transition ${
                connected
                  ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                  : "bg-rose-500/10 text-rose-300 border-rose-500/30"
              }`}
            >
              {connected ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <Wifi className="w-3.5 h-3.5 hidden sm:inline" />
                  <span>Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3.5 h-3.5" />
                  <span>Reconnecting...</span>
                </>
              )}
            </span>
          </div>
        </div>
      </header>

      {/* Main Studio Workspace */}
      <main className="max-w-[1440px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (7 cols): Camera Viewport & Dataset Training Studio */}
        <section className="lg:col-span-7 flex flex-col gap-5">
          <CameraFeed
            connected={connected}
            latestResponse={latestResponse}
            serverPrototypes={serverPrototypes}
            latestHandRef={latestHandRef}
            onFrameLandmarks={sendFrameLandmarks}
            onFpsUpdate={setFps}
            onRecordSample={async (label, hand) => {
              await recordSampleToDb(label, hand);
            }}
          />
          <DatasetStudio
            latestHandRef={latestHandRef}
            sampleCounts={sampleCounts}
            isCustomTrained={isCustomTrained}
            activeClasses={activeClasses}
            onRecordBatch={recordSamplesBatch}
            onDeleteSamples={deleteSamples}
            onTrainModel={trainCustomModel}
            onResetDemo={resetToDemoModel}
          />
        </section>

        {/* Right Column (5 cols): Hero Transcript & Telemetry */}
        <aside className="lg:col-span-5 flex flex-col gap-5">
          <TranscriptBox
            latestResponse={latestResponse}
            onSendCommand={sendCommand}
          />
          <MetricsPanel
            fps={fps}
            rttMs={rttMs}
            latestResponse={latestResponse}
            modelMetrics={modelMetrics}
          />
        </aside>
      </main>

      {/* Subtle Footer */}
      <footer className="border-t border-slate-800/60 py-3.5 px-4 sm:px-6 lg:px-8 mt-auto">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
          <span>
            SignStream v1.0 — 21×3 MediaPipe Landmarks → PyTorch MLP &amp; LSTM
          </span>
          <span className="font-mono">
            Debounce: 12/15 frames ≥85% • Cooldown: 10 IDLE frames • Target &lt;35ms
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
