import React, { useState } from "react";
import {
  Volume2,
  Delete,
  CornerDownLeft,
  Space,
  Trash2,
  Wand2,
  CheckCircle2,
  ShieldAlert,
  Copy,
  Check,
  MessageSquareText,
} from "lucide-react";
import { InferenceResponse } from "../types/signstream";

interface TranscriptBoxProps {
  latestResponse: InferenceResponse | null;
  onSendCommand: (command: string) => void;
}

export const TranscriptBox: React.FC<TranscriptBoxProps> = ({
  latestResponse,
  onSendCommand,
}) => {
  const [speaking, setSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);

  const agreement = latestResponse?.agreement_count ?? 0;
  const lockThreshold = latestResponse?.lock_threshold ?? 12;
  const windowSize = latestResponse?.window_size ?? 15;
  const lockProgressPct = Math.min(
    100,
    Math.round((agreement / lockThreshold) * 100)
  );

  const activeText = latestResponse?.active_text || "";
  const refinedSentence = latestResponse?.refined_sentence || "";
  const submitted = latestResponse?.submitted_sentences || [];

  const speakSentence = (textToSpeak: string) => {
    if (!textToSpeak.trim() || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 1.0;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  const copyOutput = async () => {
    const text = refinedSentence || activeText;
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard unavailable
    }
  };

  return (
    <div className="studio-card p-4 sm:p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/25 flex items-center justify-center">
            <MessageSquareText className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              Live Translation
              {latestResponse?.emitted_event && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  <CheckCircle2 className="w-3 h-3" />
                  {latestResponse.emitted_event}
                </span>
              )}
            </h2>
            <p className="text-[11px] text-slate-400">
              Lock-in ≥ {lockThreshold}/{windowSize} frames (≥85% conf)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={copyOutput}
            disabled={!activeText.trim()}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 border border-slate-700/80 transition"
            title="Copy translation"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Copy className="w-3.5 h-3.5 text-slate-400" />
            )}
            <span className="hidden sm:inline">{copied ? "Copied" : "Copy"}</span>
          </button>

          <button
            type="button"
            onClick={() => speakSentence(refinedSentence || activeText)}
            disabled={!activeText.trim()}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition border ${
              activeText.trim()
                ? speaking
                  ? "bg-cyan-400 text-slate-950 border-cyan-300"
                  : "bg-cyan-500/15 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/25"
                : "bg-slate-950/60 text-slate-600 border-slate-800 cursor-not-allowed"
            }`}
          >
            <Volume2 className="w-3.5 h-3.5" />
            {speaking ? "Speaking..." : "Speak"}
          </button>
        </div>
      </div>

      {/* Stability Lock-In Meter */}
      <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 font-medium">
            Hold Stability:{" "}
            <span className="text-cyan-300 font-mono font-bold">
              {latestResponse?.raw_prediction || "IDLE"}
            </span>
          </span>
          <span className="font-mono text-[11px] text-slate-300">
            {agreement}/{lockThreshold} frames ({lockProgressPct}%)
          </span>
        </div>

        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className={`h-full transition-all duration-100 rounded-full ${
              latestResponse?.status === "COOLDOWN"
                ? "bg-amber-400"
                : lockProgressPct >= 100
                ? "bg-emerald-400"
                : "bg-gradient-to-r from-cyan-500 to-sky-400"
            }`}
            style={{ width: `${lockProgressPct}%` }}
          />
        </div>

        {latestResponse?.status === "COOLDOWN" && (
          <div className="flex items-center gap-1.5 text-[11px] text-amber-300">
            <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
            <span>
              Cooldown active — hold IDLE for{" "}
              {latestResponse.cooldown_remaining} frames to repeat the same sign.
            </span>
          </div>
        )}
      </div>

      {/* Hero Output Area: Raw Buffer + Refined Grammatical Sentence */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-xl overflow-hidden flex flex-col divide-y divide-slate-800/80">
        {/* Raw Character Stream */}
        <div className="p-4 flex flex-col gap-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Character Buffer
          </span>
          <div className="min-h-[2.75rem] flex items-center">
            <span className="font-mono text-2xl font-bold tracking-wider text-white break-all">
              {activeText || (
                <span className="text-slate-600 text-sm font-sans font-normal italic">
                  Show a sign in front of the camera to begin translating...
                </span>
              )}
              <span className="inline-block w-2 h-6 ml-1 align-middle bg-cyan-400 rounded-sm animate-pulse" />
            </span>
          </div>
        </div>

        {/* Grammatical Refiner */}
        <div className="px-4 py-3 bg-emerald-500/[0.04] flex flex-col gap-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
            <Wand2 className="w-3 h-3" />
            Refined Sentence
          </span>
          <span className="text-sm font-medium text-emerald-100 min-h-[1.25rem]">
            {refinedSentence || (
              <span className="text-slate-600 text-xs italic">
                Auto-formatted sentence appears here...
              </span>
            )}
          </span>
        </div>
      </div>

      {/* Clean Quick Action Controls */}
      <div className="grid grid-cols-4 gap-2">
        <button
          type="button"
          onClick={() => onSendCommand("SPACE")}
          className="flex flex-col sm:flex-row items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl text-xs font-semibold bg-slate-950/90 hover:bg-slate-800 text-slate-200 border border-slate-800 transition"
          title="Insert Space (OPEN_PALM_HOLD 1.5s)"
        >
          <Space className="w-3.5 h-3.5 text-cyan-400" />
          <span>Space</span>
        </button>

        <button
          type="button"
          onClick={() => onSendCommand("BACKSPACE")}
          className="flex flex-col sm:flex-row items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl text-xs font-semibold bg-slate-950/90 hover:bg-slate-800 text-slate-200 border border-slate-800 transition"
          title="Delete last character (FIST_SWIPE_LEFT)"
        >
          <Delete className="w-3.5 h-3.5 text-amber-400" />
          <span>Delete</span>
        </button>

        <button
          type="button"
          onClick={() => onSendCommand("SUBMIT")}
          className="flex flex-col sm:flex-row items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition"
          title="Submit sentence (THUMBS_UP_HOLD)"
        >
          <CornerDownLeft className="w-3.5 h-3.5" />
          <span>Submit</span>
        </button>

        <button
          type="button"
          onClick={() => onSendCommand("CLEAR")}
          className="flex flex-col sm:flex-row items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 transition"
          title="Clear active buffer"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear</span>
        </button>
      </div>

      {/* Submitted History */}
      {submitted.length > 0 && (
        <div className="flex flex-col gap-2 pt-3 border-t border-slate-800/80">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Transcript History ({submitted.length})
          </span>
          <div className="flex flex-col gap-1.5 max-h-32 overflow-y-auto pr-1">
            {submitted.map((sent, i) => (
              <div
                key={`${i}-${sent}`}
                className="flex items-center justify-between gap-2 bg-slate-950/80 border border-slate-800/80 px-3 py-2 rounded-xl text-xs text-slate-200"
              >
                <span className="truncate">{sent}</span>
                <button
                  type="button"
                  onClick={() => speakSentence(sent)}
                  className="text-cyan-400 hover:text-cyan-300 p-1 rounded-md hover:bg-cyan-500/10 transition shrink-0"
                  title="Speak sentence"
                >
                  <Volume2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
