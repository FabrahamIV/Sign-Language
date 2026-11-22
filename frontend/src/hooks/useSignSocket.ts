import { useCallback, useEffect, useRef, useState } from "react";
import {
  FramePayload,
  HandLandmarks,
  InferenceResponse,
  ModelEvaluationSummary,
  Point3D,
} from "../types/signstream";

const DEFAULT_WS_URL =
  import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/sign";
const DEFAULT_API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

export function useSignSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const frameCounterRef = useRef<number>(1);
  const [connected, setConnected] = useState<boolean>(false);
  const [latestResponse, setLatestResponse] = useState<InferenceResponse | null>(
    null
  );
  const [rttMs, setRttMs] = useState<number>(0);
  const [modelMetrics, setModelMetrics] =
    useState<ModelEvaluationSummary | null>(null);
  const [serverPrototypes, setServerPrototypes] = useState<
    Record<string, Point3D[]>
  >({});
  const [sampleCounts, setSampleCounts] = useState<Record<string, number>>({});
  const [isCustomTrained, setIsCustomTrained] = useState<boolean>(false);
  const [activeClasses, setActiveClasses] = useState<string[]>([]);

  const fetchMetadata = useCallback(async () => {
    try {
      const [classesRes, metricsRes, samplesRes] = await Promise.all([
        fetch(`${DEFAULT_API_URL}/api/classes`),
        fetch(`${DEFAULT_API_URL}/api/metrics`),
        fetch(`${DEFAULT_API_URL}/api/samples`),
      ]);
      if (classesRes.ok) {
        const cData = await classesRes.json();
        if (cData.prototypes) {
          setServerPrototypes(cData.prototypes);
        }
      }
      if (metricsRes.ok) {
        const mData = await metricsRes.json();
        if (mData.model_evaluation) {
          setModelMetrics(mData.model_evaluation);
        }
      }
      if (samplesRes.ok) {
        const sData = await samplesRes.json();
        setSampleCounts(sData.sample_counts || {});
        setIsCustomTrained(Boolean(sData.is_custom_trained));
        setActiveClasses(sData.active_classes || []);
      }
    } catch {
      // Backend might still be starting up
    }
  }, []);

  useEffect(() => {
    fetchMetadata();
  }, [fetchMetadata]);

  useEffect(() => {
    let reconnectTimer: number | undefined;
    let isMounted = true;

    const connect = () => {
      if (!isMounted) return;
      try {
        const ws = new WebSocket(DEFAULT_WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isMounted) return;
          setConnected(true);
          fetchMetadata();
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const data: InferenceResponse = JSON.parse(event.data);
            if (data && typeof data.frame_id === "number") {
              const now = performance.now();
              const rtt = Math.max(0.2, now - data.client_timestamp_ms);
              setRttMs(Number(rtt.toFixed(1)));
              setLatestResponse(data);
            }
          } catch {
            // Ignore malformed message
          }
        };

        ws.onclose = () => {
          if (!isMounted) return;
          setConnected(false);
          reconnectTimer = window.setTimeout(connect, 1800);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch {
        reconnectTimer = window.setTimeout(connect, 2000);
      }
    };

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchMetadata]);

  const sendFrameLandmarks = useCallback(
    (hands: HandLandmarks[], hintLabel?: string | null) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const payload: FramePayload = {
        frame_id: frameCounterRef.current++,
        timestamp_ms: performance.now(),
        hands,
        hint_label: hintLabel || null,
      };
      ws.send(JSON.stringify(payload));
    },
    []
  );

  const sendCommand = useCallback((command: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const payload: FramePayload = {
      frame_id: frameCounterRef.current++,
      timestamp_ms: performance.now(),
      hands: [],
      command,
    };
    ws.send(JSON.stringify(payload));
  }, []);

  const recordSampleToDb = useCallback(
    async (label: string, hand: HandLandmarks) => {
      const res = await fetch(`${DEFAULT_API_URL}/api/samples`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, hand }),
      });
      const data = await res.json();
      if (data.sample_counts) {
        setSampleCounts(data.sample_counts);
      }
      return data;
    },
    []
  );

  const recordSamplesBatch = useCallback(
    async (label: string, hands: HandLandmarks[]) => {
      const res = await fetch(`${DEFAULT_API_URL}/api/samples/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, hands }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to save batch samples");
      }
      const data = await res.json();
      if (data.sample_counts) {
        setSampleCounts(data.sample_counts);
      }
      return data;
    },
    []
  );

  const deleteSamples = useCallback(async (label: string) => {
    const res = await fetch(
      `${DEFAULT_API_URL}/api/samples/${encodeURIComponent(label)}`,
      { method: "DELETE" }
    );
    const data = await res.json();
    if (data.sample_counts) {
      setSampleCounts(data.sample_counts);
    }
    return data;
  }, []);

  const trainCustomModel = useCallback(async (epochs = 45) => {
    const res = await fetch(`${DEFAULT_API_URL}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Training failed");
    }
    setIsCustomTrained(Boolean(data.is_custom_trained));
    setActiveClasses(data.active_classes || []);
    if (data.metrics) {
      setModelMetrics(data.metrics);
    }
    return data;
  }, []);

  const resetToDemoModel = useCallback(async () => {
    const res = await fetch(`${DEFAULT_API_URL}/api/train/reset-demo`, {
      method: "POST",
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Reset failed");
    }
    setIsCustomTrained(Boolean(data.is_custom_trained));
    setActiveClasses(data.active_classes || []);
    if (data.metrics) {
      setModelMetrics(data.metrics);
    }
    return data;
  }, []);

  return {
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
  };
}
