
from __future__ import annotations

import csv
import io
import os
import threading
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from flask import (  # type: ignore[import-not-found]
    Flask,
    Response,
    jsonify,
    render_template,
    send_file,
)

from database import Database
from detector import AttentionDetector, CameraWorker


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = Path(
    os.environ.get(
        "ATTENTIONAI_DB",
        str(BASE_DIR / "data" / "attentionai.db")
    )
)

SAVE_INTERVAL_SECONDS = max(
    0.5,
    float(os.environ.get("ATTENTIONAI_SAVE_INTERVAL", "2"))
)

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False


# ==========================================
# INITIALIZE COMPONENTS
# ==========================================

database = Database(DB_PATH)

detector = AttentionDetector()

camera = CameraWorker(detector)

state_lock = threading.RLock()

session_lock = threading.Lock()


session_state = {
    "active": False,
    "started_at": None,
    "session_id": None,
    "last_saved": 0.0,
    "focus_streak_started": None,
    "focus_streak_seconds": 0.0,
}


# ==========================================
# TIME HELPERS
# ==========================================

def current_duration() -> float:

    with state_lock:

        started = session_state["started_at"]

        active = session_state["active"]

    if not active or started is None:

        return 0.0

    return max(0.0, time.monotonic() - started)


# ==========================================
# SESSION SNAPSHOT
# ==========================================

def snapshot() -> dict:

    data = detector.latest()

    now = time.monotonic()

    with state_lock:

        active = session_state["active"]

        started = session_state["started_at"]

        streak_start = session_state["focus_streak_started"]

        stored_streak = session_state["focus_streak_seconds"]

    duration = (
        max(0.0, now - started)
        if active and started is not None
        else 0.0
    )

    if active and data.get("status") == "Focused":

        streak = stored_streak

        if streak_start is not None:

            streak += max(0.0, now - streak_start)

    else:

        streak = 0.0

    data.update({

        "session_active": active,

        "session_duration": round(duration, 1),

        "focus_streak": round(streak, 1),

        "server_time": datetime.now(
            timezone.utc
        ).isoformat(),

    })

    return data


# ==========================================
# FOCUS STREAK TRACKING
# ==========================================

def update_focus_streak(status: str):

    now = time.monotonic()

    with state_lock:

        if not session_state["active"]:

            return

        streak_start = session_state["focus_streak_started"]

        if status == "Focused":

            if streak_start is None:

                session_state["focus_streak_started"] = now

        else:

            if streak_start is not None:

                session_state["focus_streak_seconds"] += (
                    now - streak_start
                )

                session_state["focus_streak_started"] = None


# ==========================================
# DATABASE SAMPLE SAVING
# ==========================================

def save_sample_if_due():

    now = time.monotonic()

    with state_lock:

        if not session_state["active"]:

            return

        if (
            now - session_state["last_saved"]
            < SAVE_INTERVAL_SECONDS
        ):

            return

        session_id = session_state["session_id"]

        session_state["last_saved"] = now

    if session_id is None:

        return

    data = detector.latest()

    update_focus_streak(data.get("status", ""))

    current = snapshot()

    database.add_sample(

        session_id=session_id,

        status=current.get(
            "status",
            "No face detected"
        ),

        score=current.get(
            "attention_score",
            0
        ),

        ear=current.get("ear"),

        blink_count=current.get(
            "blink_count",
            0
        ),

        duration=current.get(
            "session_duration",
            0
        ),

        gaze=current.get(
            "gaze",
            "Unknown"
        ),

        head=current.get(
            "head_pose",
            "Unknown"
        ),

    )


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/")
def index():

    return render_template("index.html")


# ==========================================
# STATUS API
# ==========================================

@app.get("/api/status")
def api_status():

    save_sample_if_due()

    return jsonify(snapshot())


# ==========================================
# HISTORY API
# ==========================================

@app.get("/api/history")
def api_history():

    sessions = database.recent_sessions(limit=20)

    return jsonify({

        "sessions": sessions

    })


# ==========================================
# START SESSION
# ==========================================

@app.post("/api/start")
def api_start():

    with session_lock:

        with state_lock:

            if session_state["active"]:

                return jsonify({

                    "ok": True,

                    "message": "Session already running"

                })

        try:

            camera.start()

        except RuntimeError as exc:

            return jsonify({

                "ok": False,

                "error": str(exc)

            }), 503

        detector.reset()

        session_id = database.create_session()

        now = time.monotonic()

        with state_lock:

            session_state.update({

                "active": True,

                "started_at": now,

                "session_id": session_id,

                "last_saved": 0.0,

                "focus_streak_started": None,

                "focus_streak_seconds": 0.0,

            })

    return jsonify({

        "ok": True,

        "message": "Session started",

        "session_id": session_id

    })


# ==========================================
# STOP SESSION
# ==========================================

@app.post("/api/stop")
def api_stop():

    with session_lock:

        with state_lock:

            active = session_state["active"]

            session_id = session_state["session_id"]

            started = session_state["started_at"]

            duration = (

                max(0.0, time.monotonic() - started)

                if started is not None

                else 0.0

            )

            streak_start = session_state[
                "focus_streak_started"
            ]

            if streak_start is not None:

                session_state["focus_streak_seconds"] += (
                    time.monotonic() - streak_start
                )

            session_state.update({

                "active": False,

                "focus_streak_started": None,

            })

        camera.stop()

        if active and session_id is not None:

            # Save final sample before closing the session.
            data = detector.latest()

            database.add_sample(

                session_id=session_id,

                status=data.get(
                    "status",
                    "No face detected"
                ),

                score=data.get(
                    "attention_score",
                    0
                ),

                ear=data.get("ear"),

                blink_count=data.get(
                    "blink_count",
                    0
                ),

                duration=duration,

                gaze=data.get(
                    "gaze",
                    "Unknown"
                ),

                head=data.get(
                    "head_pose",
                    "Unknown"
                ),

            )

            database.finish_session(
                session_id,
                duration
            )

        with state_lock:

            session_state.update({

                "active": False,

                "started_at": None,

                "session_id": None,

                "last_saved": 0.0,

                "focus_streak_started": None,

                "focus_streak_seconds": 0.0,

            })

    return jsonify({

        "ok": True,

        "message": "Session stopped",

        "duration": round(duration, 1)

    })


# ==========================================
# RESET SESSION
# ==========================================

@app.post("/api/reset")
def api_reset():

    with session_lock:

        camera.stop()

        detector.reset()

        with state_lock:

            session_state.update({

                "active": False,

                "started_at": None,

                "session_id": None,

                "last_saved": 0.0,

                "focus_streak_started": None,

                "focus_streak_seconds": 0.0,

            })

    return jsonify({

        "ok": True,

        "message": "Session reset"

    })


# ==========================================
# CLEAR HISTORY
# ==========================================

@app.post("/api/history/clear")
def api_clear_history():

    with session_lock:

        database.clear()

    return jsonify({

        "ok": True,

        "message": "History cleared"

    })


# ==========================================
# LIVE CAMERA STREAM
# ==========================================

@app.get("/video_feed")
def video_feed():

    return Response(

        camera.mjpeg_stream(),

        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        )

    )


# ==========================================
# CSV EXPORT
# ==========================================

@app.get("/export.csv")
def export_csv():

    rows = database.export_rows()

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([

        "session_id",

        "timestamp",

        "status",

        "attention_score",

        "ear",

        "blink_count",

        "session_duration_seconds",

        "gaze",

        "head_pose"

    ])

    writer.writerows(rows)

    memory = io.BytesIO(

        output.getvalue().encode("utf-8-sig")

    )

    memory.seek(0)

    return send_file(

        memory,

        mimetype="text/csv",

        as_attachment=True,

        download_name="attentionai_sessions.csv"

    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return jsonify({

        "ok": True,

        "service": "AttentionAI",

        "camera_available": detector.latest().get(
            "camera_available",
            False
        )

    })


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=int(
            os.environ.get("PORT", "5000")
        ),

        debug=False,

        threaded=True

    )
