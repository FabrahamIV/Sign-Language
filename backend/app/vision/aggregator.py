"""Deterministic State Machine, Debounce Aggregator, and Grammar Refiner (PDD Section 6 & 7)."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import re
from typing import Deque, Dict, List, Optional, Tuple

from app.schemas import ExceptionCode, SystemStatus

# Common SIBI / BISINDO & ASL gloss/typo dictionary for optional grammar refinement
COMMON_CORRECTIONS_ID: Dict[str, str] = {
    "HALO": "Halo",
    "HALLO": "Halo",
    "SLMT": "Selamat",
    "PAGI": "pagi",
    "SIANG": "siang",
    "SORE": "sore",
    "MALAM": "malam",
    "SAYA": "saya",
    "AKU": "saya",
    "KAMU": "Anda",
    "MKN": "makan",
    "MAKAN": "makan",
    "MNM": "minum",
    "MINUM": "minum",
    "TRMKSH": "terima kasih",
    "TERIMAKASIH": "terima kasih",
    "MAKASIH": "terima kasih",
    "TLG": "tolong",
    "TOLONG": "tolong",
    "MAAF": "maaf",
    "APA": "apa",
    "KABAR": "kabar",
    "BAIK": "baik",
    "NAMA": "nama",
    "BANTU": "membantu",
    "SENANG": "senang",
    "BERTEMU": "bertemu",
}

COMMON_CORRECTIONS_EN: Dict[str, str] = {
    "HELO": "Hello",
    "HELLO": "Hello",
    "HI": "Hi",
    "THX": "thank you",
    "THANKS": "thank you",
    "PLS": "please",
    "PLEASE": "please",
    "SRY": "sorry",
    "SORRY": "sorry",
    "GD": "good",
    "GOOD": "good",
    "MRNG": "morning",
    "MORNING": "morning",
    "HOW": "how",
    "ARE": "are",
    "YOU": "you",
    "U": "you",
    "MY": "my",
    "NAME": "name",
    "IS": "is",
    "HELP": "help",
    "NEED": "need",
    "WATER": "water",
    "FOOD": "food",
}


def refine_gloss_sentence(raw_text: str, language: str = "ID") -> Tuple[str, List[str]]:
    """Reconstruct raw finger-spelled tokens or sign glosses into natural grammatical text."""
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    if not cleaned:
        return "", []

    lexicon = COMMON_CORRECTIONS_ID if language.upper() == "ID" else COMMON_CORRECTIONS_EN
    words = cleaned.split(" ")
    out_words: List[str] = []
    corrections: List[str] = []

    for w in words:
        upper_w = w.upper()
        if upper_w in lexicon:
            replacement = lexicon[upper_w]
            if replacement.lower() != w.lower():
                corrections.append(f"{w} -> {replacement}")
            out_words.append(replacement)
        else:
            out_words.append(w.lower())

    sentence = " ".join(out_words)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            question_starters = ("Apa", "Bagaimana", "Siapa", "Kapan", "Dimana", "How", "What", "Where", "When", "Who")
            if sentence.startswith(question_starters) or "kabar" in sentence.lower():
                sentence += "?"
            else:
                sentence += "."

    return sentence, corrections


@dataclass
class AggregatorStepOutput:
    """Result of feeding a single frame's prediction into the state machine."""

    status: SystemStatus
    exception_code: ExceptionCode
    exception_message: Optional[str]
    effective_prediction: Optional[str]
    effective_confidence: float
    locked_character: Optional[str]
    emitted_event: Optional[str]
    window_fill: int
    window_size: int
    agreement_count: int
    lock_threshold: int
    cooldown_remaining: int
    open_palm_hold_progress: float
    active_text: str
    refined_sentence: str
    submitted_sentences: List[str]


class SignStateAggregator:
    """Deterministic state machine for debouncing frame-level sign predictions (PDD Section 6).

    Parameters:
        window_size: Rolling window length N = 15 frames.
        confidence_threshold: Minimum softmax probability (0.85) to pass the confidence gate.
        lock_in_threshold: Minimum matching frames (12 out of 15) required to emit a class.
        idle_cooldown_frames: Number of IDLE frames (10) required before repeating the same class.
        ambiguous_frame_limit: Consecutive low-confidence frames (30) before triggering TRANSITION_NOISE.
        temporal_inertia_window: Previous valid frames (5) used during self-occlusion spikes.
        open_palm_hold_ms: Duration in ms (1500ms = 1.5s) to hold OPEN_PALM/SPACE to emit a space.
    """

    def __init__(
        self,
        window_size: int = 15,
        confidence_threshold: float = 0.85,
        lock_in_threshold: int = 12,
        idle_cooldown_frames: int = 10,
        ambiguous_frame_limit: int = 30,
        temporal_inertia_window: int = 5,
        open_palm_hold_ms: float = 1500.0,
    ) -> None:
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.lock_in_threshold = lock_in_threshold
        self.idle_cooldown_frames = idle_cooldown_frames
        self.ambiguous_frame_limit = ambiguous_frame_limit
        self.temporal_inertia_window = temporal_inertia_window
        self.open_palm_hold_ms = open_palm_hold_ms

        self.rolling_window: Deque[Optional[str]] = deque(maxlen=window_size)
        self.recent_valid_history: Deque[Tuple[str, float]] = deque(maxlen=temporal_inertia_window)
        self.wrist_x_history: Deque[Tuple[float, float]] = deque(maxlen=10)  # (timestamp_ms, wrist_x)

        self.last_emitted_class: Optional[str] = None
        self.idle_frames_since_emission: int = idle_cooldown_frames  # Ready initially
        self.low_confidence_streak: int = 0
        self.open_palm_start_ms: Optional[float] = None
        self.last_swipe_timestamp_ms: float = 0.0

        self.active_text: str = ""
        self.submitted_sentences: List[str] = []

    def reset_buffer(self) -> None:
        """Clear rolling prediction window while preserving active sentence text."""
        self.rolling_window.clear()
        self.open_palm_start_ms = None

    def apply_explicit_command(self, command: str) -> Optional[str]:
        """Execute an explicit UI button or control command ('CLEAR', 'BACKSPACE', 'SPACE', 'SUBMIT')."""
        cmd = command.strip().upper()
        if cmd == "CLEAR":
            self.active_text = ""
            self.reset_buffer()
            self.last_emitted_class = None
            self.idle_frames_since_emission = self.idle_cooldown_frames
            return "CLEAR"
        if cmd == "BACKSPACE":
            if self.active_text:
                self.active_text = self.active_text[:-1]
            return "BACKSPACE"
        if cmd == "SPACE":
            if self.active_text and not self.active_text.endswith(" "):
                self.active_text += " "
            return "SPACE"
        if cmd == "SUBMIT":
            if self.active_text.strip():
                refined, _ = refine_gloss_sentence(self.active_text.strip())
                self.submitted_sentences.append(refined or self.active_text.strip())
            self.active_text = ""
            self.reset_buffer()
            self.last_emitted_class = None
            self.idle_frames_since_emission = self.idle_cooldown_frames
            return "SUBMIT"
        return None

    def _detect_fist_swipe_left(
        self,
        predicted_label: Optional[str],
        wrist_x: float,
        timestamp_ms: float,
    ) -> bool:
        """Detect dynamic FIST_SWIPE_LEFT motion when wrist rapidly moves left across the frame."""
        self.wrist_x_history.append((timestamp_ms, wrist_x))
        if len(self.wrist_x_history) < 4:
            return False
        if timestamp_ms - self.last_swipe_timestamp_ms < 800.0:
            return False

        t0, x0 = self.wrist_x_history[0]
        dt = timestamp_ms - t0
        dx = wrist_x - x0
        # If wrist moved left by > 0.22 within 600ms while hand is in a fist-like class
        fist_like_classes = {"BACKSPACE", "FIST_SWIPE_LEFT", "A", "S", "E", "0"}
        if 50.0 <= dt <= 650.0 and dx <= -0.22 and (predicted_label in fist_like_classes):
            self.last_swipe_timestamp_ms = timestamp_ms
            self.wrist_x_history.clear()
            return True
        return False

    def step(
        self,
        predicted_label: Optional[str],
        confidence: float,
        timestamp_ms: float,
        exception_code: ExceptionCode = ExceptionCode.NONE,
        exception_message: Optional[str] = None,
        suppress_classification: bool = False,
        wrist_x: float = 0.5,
    ) -> AggregatorStepOutput:
        """Process one frame's classification output through the deterministic state machine."""
        emitted_event: Optional[str] = None
        locked_character: Optional[str] = None

        # Case 1: No Hand Detected -> Pause inference pipeline, reset rolling buffer, increment IDLE counter
        if exception_code == ExceptionCode.NO_HAND_DETECTED:
            self.rolling_window.clear()
            self.open_palm_start_ms = None
            self.low_confidence_streak = 0
            self.idle_frames_since_emission = min(
                self.idle_cooldown_frames, self.idle_frames_since_emission + 1
            )
            refined, _ = refine_gloss_sentence(self.active_text)
            return AggregatorStepOutput(
                status=SystemStatus.IDLE_WAITING,
                exception_code=exception_code,
                exception_message=exception_message,
                effective_prediction=None,
                effective_confidence=0.0,
                locked_character=None,
                emitted_event=None,
                window_fill=0,
                window_size=self.window_size,
                agreement_count=0,
                lock_threshold=self.lock_in_threshold,
                cooldown_remaining=max(0, self.idle_cooldown_frames - self.idle_frames_since_emission),
                open_palm_hold_progress=0.0,
                active_text=self.active_text,
                refined_sentence=refined,
                submitted_sentences=list(self.submitted_sentences),
            )

        # Case 2: Out-of-Bounds Hand -> Suppress classification, warn user
        if suppress_classification or exception_code == ExceptionCode.OUT_OF_BOUNDS:
            self.open_palm_start_ms = None
            refined, _ = refine_gloss_sentence(self.active_text)
            return AggregatorStepOutput(
                status=SystemStatus.WARNING,
                exception_code=exception_code,
                exception_message=exception_message,
                effective_prediction=None,
                effective_confidence=confidence,
                locked_character=None,
                emitted_event=None,
                window_fill=len(self.rolling_window),
                window_size=self.window_size,
                agreement_count=0,
                lock_threshold=self.lock_in_threshold,
                cooldown_remaining=max(0, self.idle_cooldown_frames - self.idle_frames_since_emission),
                open_palm_hold_progress=0.0,
                active_text=self.active_text,
                refined_sentence=refined,
                submitted_sentences=list(self.submitted_sentences),
            )

        # Case 3: Self-Occlusion -> Rely on temporal inertia (previous 5 valid frames)
        effective_label = predicted_label
        effective_conf = confidence
        if exception_code == ExceptionCode.SELF_OCCLUSION and self.recent_valid_history:
            inertia_labels = [lbl for lbl, _ in self.recent_valid_history]
            most_common_lbl, _ = Counter(inertia_labels).most_common(1)[0]
            effective_label = most_common_lbl
            effective_conf = max(confidence, float(self.confidence_threshold))

        # Dynamic FIST_SWIPE_LEFT detection
        if self._detect_fist_swipe_left(effective_label, wrist_x, timestamp_ms):
            if self.active_text:
                self.active_text = self.active_text[:-1]
            self.rolling_window.clear()
            emitted_event = "BACKSPACE"
            locked_character = "BACKSPACE"
            refined, _ = refine_gloss_sentence(self.active_text)
            return AggregatorStepOutput(
                status=SystemStatus.LOCKED_IN,
                exception_code=exception_code,
                exception_message=exception_message,
                effective_prediction="FIST_SWIPE_LEFT",
                effective_confidence=max(effective_conf, 0.92),
                locked_character=locked_character,
                emitted_event=emitted_event,
                window_fill=0,
                window_size=self.window_size,
                agreement_count=self.lock_in_threshold,
                lock_threshold=self.lock_in_threshold,
                cooldown_remaining=0,
                open_palm_hold_progress=0.0,
                active_text=self.active_text,
                refined_sentence=refined,
                submitted_sentences=list(self.submitted_sentences),
            )

        # Confidence Gate (< 0.85 discarded)
        if effective_label is None or effective_conf < self.confidence_threshold:
            self.rolling_window.append(None)
            self.low_confidence_streak += 1
            self.open_palm_start_ms = None

            if self.low_confidence_streak > self.ambiguous_frame_limit:
                exception_code = ExceptionCode.TRANSITION_NOISE
                exception_message = (
                    "Ambiguous gesture (TRANSITION_NOISE) — hold sign steady above 85% confidence."
                )

            refined, _ = refine_gloss_sentence(self.active_text)
            return AggregatorStepOutput(
                status=SystemStatus.WARNING
                if exception_code != ExceptionCode.NONE
                else SystemStatus.TRACKING,
                exception_code=exception_code,
                exception_message=exception_message,
                effective_prediction="TRANSITION_NOISE"
                if exception_code == ExceptionCode.TRANSITION_NOISE
                else effective_label,
                effective_confidence=effective_conf,
                locked_character=None,
                emitted_event=None,
                window_fill=len(self.rolling_window),
                window_size=self.window_size,
                agreement_count=0,
                lock_threshold=self.lock_in_threshold,
                cooldown_remaining=max(0, self.idle_cooldown_frames - self.idle_frames_since_emission),
                open_palm_hold_progress=0.0,
                active_text=self.active_text,
                refined_sentence=refined,
                submitted_sentences=list(self.submitted_sentences),
            )

        # Passed Confidence Gate (>= 0.85)
        self.low_confidence_streak = 0
        self.recent_valid_history.append((effective_label, effective_conf))

        # Handle explicit IDLE class
        if effective_label == "IDLE":
            self.rolling_window.append("IDLE")
            self.open_palm_start_ms = None
            self.idle_frames_since_emission = min(
                self.idle_cooldown_frames, self.idle_frames_since_emission + 1
            )
            refined, _ = refine_gloss_sentence(self.active_text)
            return AggregatorStepOutput(
                status=SystemStatus.IDLE_WAITING,
                exception_code=exception_code,
                exception_message=exception_message,
                effective_prediction="IDLE",
                effective_confidence=effective_conf,
                locked_character=None,
                emitted_event=None,
                window_fill=len(self.rolling_window),
                window_size=self.window_size,
                agreement_count=0,
                lock_threshold=self.lock_in_threshold,
                cooldown_remaining=max(0, self.idle_cooldown_frames - self.idle_frames_since_emission),
                open_palm_hold_progress=0.0,
                active_text=self.active_text,
                refined_sentence=refined,
                submitted_sentences=list(self.submitted_sentences),
            )

        # Track OPEN_PALM_HOLD (1.5s timer) for SPACE control gesture
        open_palm_progress = 0.0
        if effective_label in ("SPACE", "OPEN_PALM_HOLD"):
            if self.open_palm_start_ms is None:
                self.open_palm_start_ms = timestamp_ms
            elapsed_ms = max(0.0, timestamp_ms - self.open_palm_start_ms)
            open_palm_progress = min(1.0, elapsed_ms / self.open_palm_hold_ms)
            if elapsed_ms >= self.open_palm_hold_ms:
                if not self.active_text.endswith(" "):
                    self.active_text += " "
                emitted_event = "SPACE"
                locked_character = "SPACE"
                self.open_palm_start_ms = None
                self.rolling_window.clear()
                self.last_emitted_class = "SPACE"
                self.idle_frames_since_emission = 0
                refined, _ = refine_gloss_sentence(self.active_text)
                return AggregatorStepOutput(
                    status=SystemStatus.LOCKED_IN,
                    exception_code=exception_code,
                    exception_message=exception_message,
                    effective_prediction=effective_label,
                    effective_confidence=effective_conf,
                    locked_character=locked_character,
                    emitted_event=emitted_event,
                    window_fill=0,
                    window_size=self.window_size,
                    agreement_count=self.lock_in_threshold,
                    lock_threshold=self.lock_in_threshold,
                    cooldown_remaining=self.idle_cooldown_frames,
                    open_palm_hold_progress=1.0,
                    active_text=self.active_text,
                    refined_sentence=refined,
                    submitted_sentences=list(self.submitted_sentences),
                )
        else:
            self.open_palm_start_ms = None

        # Append valid prediction to 15-frame rolling window
        self.rolling_window.append(effective_label)

        # Count occurrences of non-None, non-IDLE predictions in the rolling window
        valid_preds = [p for p in self.rolling_window if p is not None and p != "IDLE"]
        if not valid_preds:
            agreement_count = 0
            dominant_class = None
        else:
            dominant_class, agreement_count = Counter(valid_preds).most_common(1)[0]

        status = SystemStatus.TRACKING
        cooldown_remaining = max(0, self.idle_cooldown_frames - self.idle_frames_since_emission)

        # Check Lock-in Threshold: >= 12 out of last 15 frames predict the exact same class C_k
        if dominant_class is not None and agreement_count >= self.lock_in_threshold:
            # Cooldown / Repeat Prevention: Block duplicate emission of C_k until 10 IDLE frames elapsed
            if (
                dominant_class == self.last_emitted_class
                and self.idle_frames_since_emission < self.idle_cooldown_frames
            ):
                status = SystemStatus.COOLDOWN
            else:
                locked_character = dominant_class
                status = SystemStatus.LOCKED_IN
                self.last_emitted_class = dominant_class
                self.idle_frames_since_emission = 0
                cooldown_remaining = self.idle_cooldown_frames

                # Execute control gesture or append character
                if dominant_class in ("SPACE", "OPEN_PALM_HOLD"):
                    if not self.active_text.endswith(" "):
                        self.active_text += " "
                    emitted_event = "SPACE"
                elif dominant_class in ("BACKSPACE", "FIST_SWIPE_LEFT"):
                    if self.active_text:
                        self.active_text = self.active_text[:-1]
                    emitted_event = "BACKSPACE"
                elif dominant_class in ("SUBMIT", "THUMBS_UP_HOLD"):
                    if self.active_text.strip():
                        refined_sub, _ = refine_gloss_sentence(self.active_text.strip())
                        self.submitted_sentences.append(refined_sub or self.active_text.strip())
                    self.active_text = ""
                    emitted_event = "SUBMIT"
                else:
                    if len(dominant_class) > 1:
                        # Multi-character custom word/gloss sign (e.g. HALO, SAYA, TERIMA_KASIH)
                        word_token = dominant_class.replace("_", " ")
                        if self.active_text and not self.active_text.endswith(" "):
                            self.active_text += " "
                        self.active_text += word_token + " "
                    else:
                        self.active_text += dominant_class
                    emitted_event = dominant_class

                self.rolling_window.clear()

        refined, _ = refine_gloss_sentence(self.active_text)
        return AggregatorStepOutput(
            status=status,
            exception_code=exception_code,
            exception_message=exception_message,
            effective_prediction=effective_label,
            effective_confidence=effective_conf,
            locked_character=locked_character,
            emitted_event=emitted_event,
            window_fill=len(self.rolling_window),
            window_size=self.window_size,
            agreement_count=agreement_count,
            lock_threshold=self.lock_in_threshold,
            cooldown_remaining=cooldown_remaining,
            open_palm_hold_progress=open_palm_progress,
            active_text=self.active_text,
            refined_sentence=refined,
            submitted_sentences=list(self.submitted_sentences),
        )

