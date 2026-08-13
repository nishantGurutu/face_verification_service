"""SQLite-backed embedding store for the face verification service."""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Optional, Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.environ.get("FACE_DB_PATH", os.path.join(DATA_DIR, "faces.db"))

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _init() -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                employee_id TEXT PRIMARY KEY,
                name TEXT,
                embedding BLOB NOT NULL,
                registered_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


_init()


def upsert_embedding(employee_id: str, name: Optional[str], embedding: np.ndarray) -> None:
    """Insert or replace the stored embedding for an employee."""
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO faces (employee_id, name, embedding) VALUES (?, ?, ?) "
            "ON CONFLICT(employee_id) DO UPDATE SET name=excluded.name, embedding=excluded.embedding, "
            "registered_at=datetime('now')",
            (employee_id, name, blob),
        )


def get_embedding(employee_id: str) -> Optional[Tuple[str, np.ndarray]]:
    """Return (name, embedding) for an employee, or None if not registered."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT name, embedding FROM faces WHERE employee_id = ?", (employee_id,)
        ).fetchone()
    if row is None:
        return None
    name, blob = row
    embedding = np.frombuffer(blob, dtype=np.float32).copy()
    return name, embedding


def count_faces() -> int:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM faces").fetchone()
    return int(row[0]) if row else 0
