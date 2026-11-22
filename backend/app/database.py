"""SQLite database persistence for custom recorded gesture datasets, session logs, and model metrics."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "signstream.db"


class SignStreamDatabase:
    """Lightweight SQLite manager for SignStream persistence."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        env_path = os.getenv("SIGNSTREAM_DB_PATH")
        self.db_path = Path(db_path or env_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gesture_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    handedness TEXT NOT NULL DEFAULT 'Right',
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    emitted_token TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    active_text TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    num_classes INTEGER NOT NULL,
                    val_accuracy REAL NOT NULL,
                    val_f1_macro REAL NOT NULL,
                    test_accuracy REAL NOT NULL,
                    test_f1_macro REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def insert_gesture_sample(
        self, label: str, handedness: str, normalized_features: List[float]
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO gesture_samples (label, handedness, features_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (label.strip().upper(), handedness, json.dumps(normalized_features), now),
            )
            return int(cur.lastrowid or 0)

    def insert_gesture_samples_batch(
        self, label: str, samples: List[tuple[str, List[float]]]
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        clean_label = label.strip().upper()
        rows = [
            (clean_label, handedness, json.dumps(feats), now)
            for handedness, feats in samples
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO gesture_samples (label, handedness, features_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_all_gesture_samples(self) -> List[tuple[str, List[float]]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT label, features_json FROM gesture_samples ORDER BY id ASC"
            ).fetchall()
            return [
                (str(row["label"]), json.loads(row["features_json"]))
                for row in rows
            ]

    def delete_samples_for_label(self, label: str) -> int:
        clean_label = label.strip().upper()
        with self._connect() as conn:
            if clean_label == "ALL":
                cur = conn.execute("DELETE FROM gesture_samples")
            else:
                cur = conn.execute(
                    "DELETE FROM gesture_samples WHERE label = ?", (clean_label,)
                )
            return int(cur.rowcount or 0)

    def log_session_emission(
        self,
        session_id: str,
        emitted_token: str,
        confidence: float,
        active_text: str,
        latency_ms: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_logs (session_id, emitted_token, confidence, active_text, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, emitted_token, confidence, active_text, latency_ms, now),
            )

    def save_model_evaluation(self, summary: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_evaluations (
                    model_name, num_classes, val_accuracy, val_f1_macro,
                    test_accuracy, test_f1_macro, metrics_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.get("model_name", "StaticSignMLP"),
                    int(summary.get("num_classes", 40)),
                    float(summary.get("val_accuracy", 0.0)),
                    float(summary.get("val_f1_macro", 0.0)),
                    float(summary.get("test_accuracy", 0.0)),
                    float(summary.get("test_f1_macro", 0.0)),
                    json.dumps(summary),
                    now,
                ),
            )

    def get_latest_evaluation(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metrics_json FROM model_evaluations ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                return json.loads(row["metrics_json"])
            return None

    def get_sample_counts(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) as cnt FROM gesture_samples GROUP BY label ORDER BY label"
            ).fetchall()
            return {row["label"]: int(row["cnt"]) for row in rows}

    def get_recent_logs(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

