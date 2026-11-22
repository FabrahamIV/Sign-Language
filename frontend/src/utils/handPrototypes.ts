import { Point3D } from "../types/signstream";

/**
 * Standard MediaPipe Hands 21-keypoint bone connections for HTML5 <canvas> skeletal rendering.
 */
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index Finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle Finger
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring Finger
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky Finger
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

export const ALPHABET_CLASSES = Array.from({ length: 26 }, (_, i) =>
  String.fromCharCode(65 + i)
);
export const DIGIT_CLASSES = Array.from({ length: 10 }, (_, i) => String(i));
export const CONTROL_CLASSES = ["IDLE", "SPACE", "BACKSPACE", "SUBMIT"];

type CurlConfig = [number, number, number, number, number];
type SpreadConfig = [number, number, number, number, number];

function buildPose(
  curls: CurlConfig,
  spread: SpreadConfig = [-0.12, -0.06, 0.0, 0.06, 0.12],
  thumbMode: "side" | "up" | "tucked" | "across" = "side",
  palmTiltZ = 0.0,
  rotAngle = 0.0
): Point3D[] {
  const pts: Point3D[] = Array.from({ length: 21 }, () => ({
    x: 0.5,
    y: 0.72,
    z: 0.0,
    visibility: 0.99,
  }));

  pts[0] = { x: 0.5, y: 0.72, z: 0.0, visibility: 0.99 };
  const tCurl = curls[0];
  const tSpread = spread[0];

  if (thumbMode === "up") {
    pts[1] = { x: 0.48, y: 0.64, z: -0.02, visibility: 0.99 };
    pts[2] = { x: 0.47, y: 0.54, z: -0.03, visibility: 0.99 };
    pts[3] = { x: 0.47, y: 0.44, z: -0.03, visibility: 0.99 };
    pts[4] = { x: 0.47, y: 0.34, z: -0.04, visibility: 0.99 };
  } else if (thumbMode === "tucked") {
    pts[1] = { x: 0.43, y: 0.65, z: -0.02, visibility: 0.99 };
    pts[2] = { x: 0.44, y: 0.58, z: -0.03, visibility: 0.99 };
    pts[3] = { x: 0.49, y: 0.56, z: -0.04, visibility: 0.99 };
    pts[4] = { x: 0.54, y: 0.57, z: -0.04, visibility: 0.99 };
  } else if (thumbMode === "across") {
    pts[1] = { x: 0.43, y: 0.65, z: -0.02, visibility: 0.99 };
    pts[2] = { x: 0.45, y: 0.58, z: -0.04, visibility: 0.99 };
    pts[3] = { x: 0.51, y: 0.54, z: -0.06, visibility: 0.99 };
    pts[4] = { x: 0.56, y: 0.53, z: -0.06, visibility: 0.99 };
  } else {
    const ext = 1.0 - 0.55 * tCurl;
    pts[1] = { x: 0.42 + tSpread * 0.2, y: 0.65, z: -0.01, visibility: 0.99 };
    pts[2] = { x: 0.36 + tSpread * 0.4, y: 0.58, z: -0.02, visibility: 0.99 };
    pts[3] = {
      x: 0.31 * ext + 0.44 * (1 - ext),
      y: 0.52 + 0.06 * tCurl,
      z: -0.03,
      visibility: 0.99,
    };
    pts[4] = {
      x: 0.26 * ext + 0.46 * (1 - ext),
      y: 0.46 + 0.1 * tCurl,
      z: -0.04,
      visibility: 0.99,
    };
  }

  const mcpXs = [0.43, 0.48, 0.53, 0.58];
  const mcpYs = [0.54, 0.52, 0.54, 0.57];
  const fingerLens = [0.22, 0.25, 0.23, 0.18];

  for (let f = 0; f < 4; f++) {
    const base = 5 + f * 4;
    const curl = curls[f + 1];
    const sp = spread[f + 1];
    const mx = mcpXs[f];
    const my = mcpYs[f];
    const flen = fingerLens[f];

    pts[base] = { x: mx, y: my, z: palmTiltZ * 0.2, visibility: 0.99 };
    const pipY = my - flen * 0.42 * (1.0 - 0.35 * curl);
    const dipY = my - flen * 0.75 * (1.0 - 0.85 * curl);
    const tipY = my - flen * 1.0 * (1.0 - 1.15 * curl);

    const zFold = -0.06 * curl + palmTiltZ;
    pts[base + 1] = { x: mx + sp * 0.35, y: pipY, z: zFold * 0.5, visibility: 0.99 };
    pts[base + 2] = { x: mx + sp * 0.7, y: dipY, z: zFold * 0.85, visibility: 0.99 };
    pts[base + 3] = { x: mx + sp * 1.0, y: tipY, z: zFold, visibility: 0.99 };
  }

  if (Math.abs(rotAngle) > 1e-4) {
    const c = Math.cos(rotAngle);
    const s = Math.sin(rotAngle);
    const ox = pts[0].x;
    const oy = pts[0].y;
    for (let i = 1; i < 21; i++) {
      const rx = pts[i].x - ox;
      const ry = pts[i].y - oy;
      pts[i].x = Math.min(0.95, Math.max(0.05, ox + rx * c - ry * s));
      pts[i].y = Math.min(0.95, Math.max(0.05, oy + rx * s + ry * c));
    }
  }

  return pts;
}

const LOCAL_PROTOTYPES: Record<string, Point3D[]> = {
  A: buildPose([0.05, 0.95, 0.95, 0.95, 0.95], undefined, "side"),
  B: buildPose([0.9, 0.0, 0.0, 0.0, 0.0], [-0.05, -0.02, 0.0, 0.02, 0.04], "tucked"),
  C: buildPose([0.35, 0.45, 0.45, 0.45, 0.45], [-0.14, -0.05, 0.0, 0.05, 0.1], "side", -0.08),
  D: buildPose([0.55, 0.0, 0.85, 0.85, 0.85], undefined, "tucked"),
  E: buildPose([0.85, 0.8, 0.8, 0.8, 0.8], undefined, "tucked", -0.04),
  F: buildPose([0.45, 0.65, 0.0, 0.0, 0.0], [-0.06, -0.03, 0.02, 0.07, 0.12], "tucked"),
  G: buildPose([0.15, 0.0, 0.95, 0.95, 0.95], undefined, "side", 0, -1.15),
  H: buildPose([0.8, 0.0, 0.0, 0.95, 0.95], [-0.05, -0.02, 0.01, 0.04, 0.08], "side", 0, -1.15),
  I: buildPose([0.8, 0.95, 0.95, 0.95, 0.0], undefined, "across"),
  J: buildPose([0.8, 0.95, 0.95, 0.95, 0.05], undefined, "across", 0, 0.45),
  K: buildPose([0.2, 0.0, 0.08, 0.95, 0.95], [-0.04, -0.06, 0.06, 0.05, 0.08], "side", -0.05),
  L: buildPose([0.0, 0.0, 0.95, 0.95, 0.95], [-0.22, -0.04, 0.0, 0.04, 0.08], "side"),
  M: buildPose([0.9, 0.72, 0.72, 0.72, 0.98], undefined, "across", 0.05),
  N: buildPose([0.85, 0.7, 0.7, 0.96, 0.98], undefined, "across", -0.05),
  O: buildPose([0.5, 0.62, 0.62, 0.62, 0.62], [-0.06, -0.02, 0.0, 0.02, 0.04], "tucked"),
  P: buildPose([0.25, 0.0, 0.15, 0.95, 0.95], undefined, "side", 0, -1.65),
  Q: buildPose([0.1, 0.1, 0.95, 0.95, 0.95], undefined, "side", 0, -1.85),
  R: buildPose([0.85, 0.0, 0.0, 0.95, 0.95], [-0.06, 0.04, -0.03, 0.05, 0.09], "tucked"),
  S: buildPose([0.75, 0.98, 0.98, 0.98, 0.98], undefined, "across"),
  T: buildPose([0.4, 0.78, 0.95, 0.95, 0.95], undefined, "tucked", 0, 0.2),
  U: buildPose([0.85, 0.0, 0.0, 0.95, 0.95], [-0.06, -0.01, 0.01, 0.05, 0.09], "tucked"),
  V: buildPose([0.85, 0.0, 0.0, 0.95, 0.95], [-0.06, -0.11, 0.09, 0.05, 0.09], "tucked"),
  W: buildPose([0.85, 0.0, 0.0, 0.0, 0.95], [-0.06, -0.1, 0.0, 0.1, 0.09], "tucked"),
  X: buildPose([0.85, 0.42, 0.95, 0.95, 0.95], undefined, "tucked"),
  Y: buildPose([0.0, 0.95, 0.95, 0.95, 0.0], [-0.2, -0.04, 0.0, 0.04, 0.18], "side"),
  Z: buildPose([0.85, 0.05, 0.95, 0.95, 0.95], undefined, "tucked", 0, -0.45),
  "0": buildPose([0.52, 0.58, 0.58, 0.58, 0.58], [-0.08, -0.03, 0.0, 0.03, 0.06], "side", 0.06),
  "1": buildPose([0.85, 0.0, 0.95, 0.95, 0.95], [-0.06, 0.0, 0.0, 0.04, 0.08], "tucked"),
  "2": buildPose([0.85, 0.0, 0.0, 0.95, 0.95], [-0.06, -0.08, 0.06, 0.05, 0.09], "side", 0.05),
  "3": buildPose([0.0, 0.0, 0.0, 0.95, 0.95], [-0.16, -0.06, 0.04, 0.05, 0.09], "side"),
  "4": buildPose([0.9, 0.0, 0.0, 0.0, 0.0], [-0.06, -0.11, -0.03, 0.05, 0.13], "tucked"),
  "5": buildPose([0.0, 0.0, 0.0, 0.0, 0.0], [-0.18, -0.1, -0.02, 0.06, 0.14], "side"),
  "6": buildPose([0.55, 0.0, 0.0, 0.0, 0.85], [-0.06, -0.08, 0.0, 0.08, 0.04], "across"),
  "7": buildPose([0.55, 0.0, 0.0, 0.85, 0.0], [-0.06, -0.08, 0.0, 0.02, 0.12], "across"),
  "8": buildPose([0.55, 0.0, 0.85, 0.0, 0.0], [-0.06, -0.08, 0.0, 0.06, 0.12], "across"),
  "9": buildPose([0.55, 0.82, 0.0, 0.0, 0.0], [-0.06, -0.02, 0.02, 0.07, 0.12], "across"),
  IDLE: buildPose([0.3, 0.32, 0.35, 0.38, 0.4], [-0.06, -0.03, 0.0, 0.03, 0.06], "side", 0, 2.45),
  SPACE: buildPose([0.0, 0.0, 0.0, 0.0, 0.0], [-0.22, -0.13, 0.0, 0.13, 0.24], "side", 0.05),
  BACKSPACE: buildPose([0.92, 0.98, 0.98, 0.98, 0.98], undefined, "across", 0, -0.65),
  SUBMIT: buildPose([0.0, 0.96, 0.96, 0.96, 0.96], undefined, "up", 0, 1.35),
};

export function getSimulatedLandmarks(
  label: string,
  serverPrototypes?: Record<string, Point3D[]>,
  edgeScenario: string = "NONE"
): Point3D[] {
  const source = serverPrototypes?.[label] || LOCAL_PROTOTYPES[label] || LOCAL_PROTOTYPES.A;
  const t = performance.now() * 0.004;

  return source.map((pt, idx) => {
    const microX = Math.sin(t + idx * 0.7) * 0.0022;
    const microY = Math.cos(t * 1.1 + idx * 0.5) * 0.0022;
    let x = pt.x + microX;
    let y = pt.y + microY;
    let z = pt.z;
    let visibility = 0.99;

    if (edgeScenario === "OUT_OF_BOUNDS" && idx === 8) {
      x = 0.992; // Triggers Out-of-Bounds (<0.02 or >0.98)
    } else if (edgeScenario === "LOW_LIGHTING_BLUR") {
      visibility = 0.32; // Triggers Low Lighting (<0.50)
    } else if (edgeScenario === "SELF_OCCLUSION" && idx === 4) {
      z = 1.45; // Triggers Z-depth spike (>0.25 std)
    }

    return { x, y, z, visibility };
  });
}

