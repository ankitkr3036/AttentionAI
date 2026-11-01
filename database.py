from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ended_at TEXT,
                    duration_seconds REAL NOT NULL DEFAULT 0
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    attention_score INTEGER NOT NULL,
                    ear REAL,
                    blink_count INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    gaze TEXT,
                    head_pose TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)

    def create_session(self) -> int:
        with self.connect() as con:
            cursor = con.execute("INSERT INTO sessions DEFAULT VALUES")
            return int(cursor.lastrowid)

    def add_sample(self, session_id, status, score, ear, blink_count,
                duration, gaze, head):
        with self.connect() as con:
            con.execute("""
                INSERT INTO samples
                (session_id, status, attention_score, ear, blink_count,
                duration_seconds, gaze, head_pose)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, status, int(score), ear, int(blink_count),
                float(duration), gaze, head))

    def finish_session(self, session_id, duration):
        with self.connect() as con:
            con.execute("UPDATE sessions SET ended_at=CURRENT_TIMESTAMP, duration_seconds=? WHERE id=?",
                        (float(duration), session_id))

    def recent_sessions(self, limit=20):
        with self.connect() as con:
            rows = con.execute("""
                SELECT s.id, s.started_at, s.ended_at, s.duration_seconds,
                    COUNT(p.id) AS sample_count,
                    COALESCE(ROUND(AVG(p.attention_score)), 0) AS average_score,
                    COALESCE(MAX(p.blink_count), 0) AS blink_count
                FROM sessions s LEFT JOIN samples p ON p.session_id=s.id
                GROUP BY s.id ORDER BY s.id DESC LIMIT ?
            """, (int(limit),)).fetchall()
            return [dict(row) for row in rows]

    def export_rows(self):
        with self.connect() as con:
            rows = con.execute("""
                SELECT session_id, timestamp, status, attention_score, ear,
                    blink_count, duration_seconds, gaze, head_pose
                FROM samples ORDER BY id
            """).fetchall()
            return [tuple(row) for row in rows]

    def clear(self):
        with self.connect() as con:
            con.execute("DELETE FROM samples")
            con.execute("DELETE FROM sessions")
