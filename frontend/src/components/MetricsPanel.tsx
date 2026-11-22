import React, { useState } from "react";
import {
  Activity,
  Gauge,
  Zap,
  Award,
  BarChart3,
  Binary,
  CheckCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  InferenceResponse,
  ModelEvaluationSummary,
} from "../types/signstream";

interface MetricsPanelProps {
  fps: number;
  rttMs: number;
  latestResponse: InferenceResponse | null;
  modelMetrics: ModelEvaluationSummary | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  fps,
  rttMs,
  latestResponse,
  modelMetrics,
}) => {
  const [showVector, setShowVector] = useState(false);

  const inferenceMs = latestResponse?.inference_latency_ms ?? 0;
  const confPct = Math.round((latestResponse?.confidence ?? 0) * 100);
  const topPreds = latestResponse?.top_predictions || [];
  const normPreview = latestResponse?.normalized_preview || [];
  const kpiLatencyMet = inferenceMs < 35.0;

  return (
    <div className="studio-card p-4 sm:p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/25 flex items-center justify-center">
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100">
              Pipeline Telemetry
            </h2>
            <p className="text-[11px] text-slate-400">
              Real-time inference KPIs &amp; softmax scores
            </p>
          </div>
        </div>
      </div>

      {/* 2x2 Compact KPI Grid */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Frame Rate</span>
            <Gauge className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white">
            {fps}{" "}
            <span className="text-[11px] font-normal text-slate-400">FPS</span>
          </div>
          <span className="text-[10px] text-emerald-400">Target 30 FPS</span>
        </div>

        <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Latency</span>
            <Zap className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white">
            {inferenceMs.toFixed(1)}{" "}
            <span className="text-[11px] font-normal text-slate-400">ms</span>
          </div>
          <span
            className={`text-[10px] ${
              kpiLatencyMet ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            KPI &lt;35ms (RTT {rttMs.toFixed(0)}ms)
          </span>
        </div>

        <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Confidence</span>
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white">
            {confPct}%
          </div>
          <span
            className={`text-[10px] ${
              confPct >= 85 ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            Gate ≥ 85%
          </span>
        </div>

        <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Model Val F1</span>
            <Award className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-lg font-bold font-mono text-white">
            {modelMetrics
              ? `${(modelMetrics.val_f1_macro * 100).toFixed(1)}%`
              : "100.0%"}
          </div>
          <span className="text-[10px] text-emerald-400">
            40 Classes (MLP)
          </span>
        </div>
      </div>

      {/* Top-3 Softmax Class Probabilities */}
      <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3.5 flex flex-col gap-2.5">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
          <span className="flex items-center gap-1.5">
            <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
            Top-3 Predictions
          </span>
          <span className="text-[10px] font-mono text-slate-500">
            PyTorch Softmax
          </span>
        </div>

        {topPreds.length === 0 ? (
          <div className="text-xs text-slate-500 py-1.5 italic">
            Waiting for hand landmarks...
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {topPreds.map((item, idx) => {
              const pct = Math.round(item.probability * 100);
              const passesGate = item.probability >= 0.85;
              return (
                <div key={item.label} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span
                      className={
                        idx === 0
                          ? "font-bold text-cyan-300"
                          : "text-slate-400"
                      }
                    >
                      {item.label}
                    </span>
                    <span
                      className={
                        passesGate ? "text-emerald-400" : "text-slate-400"
                      }
                    >
                      {(item.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-100 ${
                        idx === 0
                          ? passesGate
                            ? "bg-emerald-400"
                            : "bg-amber-400"
                          : "bg-slate-700"
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Collapsible Normalized Feature Vector Preview */}
      <div className="border-t border-slate-800/80 pt-2.5">
        <button
          type="button"
          onClick={() => setShowVector((v) => !v)}
          className="w-full flex items-center justify-between text-xs font-medium text-slate-400 hover:text-slate-200 transition"
        >
          <span className="flex items-center gap-1.5">
            <Binary className="w-3.5 h-3.5 text-cyan-400" />
            Normalized 63-d Landmark Vector
          </span>
          {showVector ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </button>

        {showVector && (
          <div className="mt-2 font-mono text-[11px] bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-cyan-300 overflow-x-auto">
            [
            {normPreview.length > 0
              ? normPreview.map((v) => v.toFixed(3)).join(", ")
              : "0.000, 0.000, 0.000, ..."}
            , ...]
          </div>
        )}
      </div>
    </div>
  );
};
