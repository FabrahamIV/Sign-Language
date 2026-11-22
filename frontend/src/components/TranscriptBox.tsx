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
  Hand,
} from "lucide-react";
import { InferenceResponse } from "../types/signstream";

interface TranscriptBoxProps {
  latestResponse: InferenceResponse | null;
  onSendCommand: (command: string) => void;
}

const GESTURE_FRIENDLY_MAP: Record<string, { icon: string; label: string }> = {
  HALO: { icon: "✋", label: "HALO" },
  BAGUS: { icon: "👍", label: "BAGUS" },
  PEACE: { icon: "✌️", label: "PEACE" },
  I_LOVE_YOU: { icon: "🤟", label: "I LOVE YOU" },
  OKE: { icon: "👌", label: "OKE" },
  "1": { icon: "☝️", label: "SATU (1)" },
  L: { icon: "👆", label: "L" },
  B: { icon: "🖐️", label: "B" },
  W: { icon: "🤟", label: "W" },
  Y: { icon: "🤙", label: "Y" },
  I: { icon: "🤙", label: "I" },
  A: { icon: "✊", label: "A" },
};

const QUICK_GUIDE = [
  "✋ HALO",
  "👍 BAGUS",
  "✌️ PEACE",
  "🤟 I LOVE YOU",
  "👌 OKE",
  "☝️ 1",
  "✊ A",
];

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

  const rawPred = latestResponse?.raw_prediction || null;
  const confPct = Math.round((latestResponse?.confidence ?? 0) * 100);
  const friendlyGesture = rawPred
    ? GESTURE_FRIENDLY_MAP[rawPred] || {
        icon: "🤚",
        label: rawPred.replace(/_/g, " "),
      }
    : null;

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
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/25 flex items-center justify-center">
            <MessageSquareText className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              Real-Time Gesture Translator
              {latestResponse?.emitted_event && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  <CheckCircle2 className="w-3 h-3" />
                  Added: {latestResponse.emitted_event.replace(/_/g, " ")}
                </span>
              )}
            </h2>
            <p className="text-[11px] text-slate-400">
              Live hand gesture detection &amp; translated word box
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

      {/* HERO LIVE GESTURE DISPLAY (Instant Real-Time Feedback beside Camera) */}
      <div className="bg-gradient-to-br from-slate-950 via-slate-900/90 to-cyan-950/30 border border-cyan-500/25 rounded-xl p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
            <Hand className="w-3.5 h-3.5" />
            Current Hand Gesture (Real-Time)
          </span>
          <span className="text-xs font-mono font-semibold text-slate-300">
            {rawPred ? `${confPct}% confidence` : "No hand detected"}
          </span>
        </div>

        <div className="flex items-center justify-between gap-3 py-1">
          {friendlyGesture ? (
            <div className="flex items-center gap-3">
              <span className="text-4xl select-none" role="img" aria-label={friendlyGesture.label}>
                {friendlyGesture.icon}
              </span>
              <div className="flex flex-col">
                <span className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white font-mono">
                  {friendlyGesture.label}
                </span>
                <span className="text-[11px] text-slate-400">
                  Hold steady (~0.4s) to lock into text box below
                </span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3 text-slate-500 py-1">
              <span className="text-3xl opacity-50">👋</span>
              <div className="flex flex-col">
                <span className="text-lg font-semibold text-slate-400">
                  Waiting for your hand...
                </span>
                <span className="text-[11px] text-slate-500">
                  Show any gesture in front of the camera
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Hold Stability Lock-In Bar */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400">
              Lock-in progress ({agreement}/{lockThreshold} frames)
            </span>
            <span className="font-mono text-cyan-300">{lockProgressPct}%</span>
          </div>
          <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
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
        </div>

        {latestResponse?.status === "COOLDOWN" && (
          <div className="flex items-center gap-1.5 text-[11px] text-amber-300">
            <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
            <span>
              Word locked! Lower your hand briefly ({latestResponse.cooldown_remaining} frames) or change gesture for the next word.
            </span>
          </div>
        )}
      </div>

      {/* REAL-TIME TRANSLATED TEXT BOX */}
      <div className="flex flex-col gap-1.5">
        <label
          htmlFor="translated-output-box"
          className="text-[11px] font-semibold uppercase tracking-wider text-slate-400"
        >
          Translated Words &amp; Text Output
        </label>
        <textarea
          id="translated-output-box"
          readOnly
          rows={3}
          value={activeText}
          placeholder="Your translated words will appear here in real time as you sign..."
          className="w-full bg-slate-950/90 border border-slate-700/80 rounded-xl px-4 py-3 font-mono text-lg font-bold tracking-wide text-white placeholder:text-slate-600 placeholder:font-sans placeholder:font-normal placeholder:text-sm focus:outline-none focus:border-cyan-500 resize-none"
        />
      </div>

      {/* Grammatical Sentence Refiner */}
      <div className="px-3.5 py-2.5 rounded-xl bg-emerald-500/[0.06] border border-emerald-500/25 flex flex-col gap-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
          <Wand2 className="w-3 h-3" />
          Formatted Sentence
        </span>
        <span className="text-sm font-medium text-emerald-100 min-h-[1.25rem]">
          {refinedSentence || (
            <span className="text-slate-500 text-xs italic">
              Auto-formatted sentence appears here...
            </span>
          )}
        </span>
      </div>

      {/* Quick Action Buttons */}
      <div className="grid grid-cols-4 gap-2">
        <button
          type="button"
          onClick={() => onSendCommand("SPACE")}
          className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-xl text-xs font-semibold bg-slate-950/90 hover:bg-slate-800 text-slate-200 border border-slate-800 transition"
        >
          <Space className="w-3.5 h-3.5 text-cyan-400" />
          <span>Space</span>
        </button>

        <button
          type="button"
          onClick={() => onSendCommand("BACKSPACE")}
          className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-xl text-xs font-semibold bg-slate-950/90 hover:bg-slate-800 text-slate-200 border border-slate-800 transition"
        >
          <Delete className="w-3.5 h-3.5 text-amber-400" />
          <span>Delete</span>
        </button>

        <button
          type="button"
          onClick={() => onSendCommand("SUBMIT")}
          className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-xl text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition"
        >
          <CornerDownLeft className="w-3.5 h-3.5" />
          <span>Submit</span>
        </button>

        <button
          type="button"
          onClick={() => onSendCommand("CLEAR")}
          className="flex items-center justify-center gap-1.5 py-2 px-2 rounded-xl text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 transition"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear</span>
        </button>
      </div>

      {/* Built-in Try-Now Gesture Cheat Sheet */}
      <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-800/80">
        <span className="text-[11px] text-slate-400 mr-1">Try on Camera:</span>
        {QUICK_GUIDE.map((item) => (
          <span
            key={item}
            className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-slate-950 text-slate-300 border border-slate-800"
          >
            {item}
          </span>
        ))}
      </div>

      {/* Submitted History */}
      {submitted.length > 0 && (
        <div className="flex flex-col gap-2 pt-2 border-t border-slate-800/80">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Transcript History ({submitted.length})
          </span>
          <div className="flex flex-col gap-1.5 max-h-28 overflow-y-auto pr-1">
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
