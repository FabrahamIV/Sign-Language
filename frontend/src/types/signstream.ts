export type SystemStatus =
  | "IDLE_WAITING"
  | "TRACKING"
  | "LOCKED_IN"
  | "COOLDOWN"
  | "WARNING";

export type ExceptionCode =
  | "NONE"
  | "NO_HAND_DETECTED"
  | "LOW_LIGHTING_BLUR"
  | "OUT_OF_BOUNDS"
  | "TRANSITION_NOISE"
  | "SELF_OCCLUSION";

export interface Point3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface HandLandmarks {
  handedness: string;
  score: number;
  landmarks: Point3D[];
}

export interface FramePayload {
  frame_id: number;
  timestamp_ms: number;
  hands: HandLandmarks[];
  hint_label?: string | null;
  command?: string | null;
}

export interface TopPrediction {
  label: string;
  probability: number;
}

export interface InferenceResponse {
  frame_id: number;
  client_timestamp_ms: number;
  inference_latency_ms: number;
  status: SystemStatus;
  exception_code: ExceptionCode;
  exception_message?: string | null;
  raw_prediction?: string | null;
  confidence: number;
  top_predictions: TopPrediction[];
  dynamic_token?: string | null;
  locked_character?: string | null;
  emitted_event?: string | null;
  window_fill: number;
  window_size: number;
  agreement_count: number;
  lock_threshold: number;
  cooldown_remaining: number;
  open_palm_hold_progress: number;
  active_text: string;
  refined_sentence: string;
  submitted_sentences: string[];
  normalized_preview: number[];
}

export interface ModelEvaluationSummary {
  model_name: string;
  num_classes: number;
  classes: string[];
  val_accuracy: number;
  val_f1_macro: number;
  test_accuracy: number;
  test_f1_macro: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  trained_at: string;
}

