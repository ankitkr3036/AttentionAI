
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


# MediaPipe Face Mesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio."""

    points = np.asarray(points, dtype=np.float64)

    if points.shape != (6, 2):
        raise ValueError("EAR expects an array with shape (6, 2)")

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal <= 1e-9:
        return 0.0

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


def classify_attention(
    face_visible: bool,
    ear: Optional[float],
    closed_seconds: float,
    gaze: str,
    head_pose: str,
    ear_threshold: float = 0.21,
    drowsy_seconds: float = 1.5
) -> str:

    if not face_visible:
        return "No face detected"

    if (
        ear is not None
        and ear < ear_threshold
        and closed_seconds >= drowsy_seconds
    ):
        return "Drowsy"

    away = (
        gaze in ("Left", "Right")
        or head_pose in (
            "Turned left",
            "Turned right",
            "Lowered"
        )
    )

    return "Distracted" if away else "Focused"


def calculate_attention_score(
    gaze: str,
    head_pose: str,
    ear: Optional[float],
    closed_seconds: float,
    ear_threshold: float = 0.21,
    drowsy_seconds: float = 1.5
) -> int:
    """
    Calculate heuristic attention score from 0 to 100.

    100 = Eyes open, centered gaze, forward head.
    """

    score = 100

    # Gaze detection
    if gaze in ("Left", "Right"):
        score -= 20

    elif gaze in ("Unknown", "Uncertain"):
        score -= 5

    # Head pose detection
    if head_pose in ("Turned left", "Turned right"):
        score -= 20

    elif head_pose == "Lowered":
        score -= 25

    elif head_pose in ("Unknown", "Uncertain"):
        score -= 5

    # Eye closure detection
    if ear is not None and ear < ear_threshold:

        if closed_seconds >= drowsy_seconds:
            score -= 50

        else:
            score -= 10

    return max(0, min(100, score))


class AttentionDetector:

    def __init__(
        self,
        ear_threshold: float = 0.21,
        drowsy_seconds: float = 1.5
    ):

        self.ear_threshold = ear_threshold
        self.drowsy_seconds = drowsy_seconds

        self._lock = threading.Lock()

        self._mesh = None
        self._closed_since = None
        self._eye_was_closed = False

        self._blink_count = 0

        self._gaze_history = deque(maxlen=7)
        self._head_history = deque(maxlen=7)

        self._latest = self._empty_status()
        self._last_frame = None

    @staticmethod
    def _empty_status():

        return {
            "status": "No face detected",
            "attention_score": 0,
            "ear": None,
            "blink_count": 0,
            "gaze": "Unknown",
            "head_pose": "Unknown",
            "eyes_closed_seconds": 0.0,
            "alert": False,
            "face_detected": False,
            "camera_available": False,
            "message": "Press Start to connect your camera."
        }

    def _get_mesh(self):

        if mp is None:
            raise RuntimeError(
                "MediaPipe is not installed. "
                "Run: pip install -r requirements.txt"
            )

        if self._mesh is None:

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        return self._mesh

    @staticmethod
    def _smooth(values, fallback="Unknown"):

        if not values:
            return fallback

        return max(set(values), key=values.count)

    def process(self, frame: np.ndarray) -> np.ndarray:

        if frame is None or frame.size == 0:
            raise ValueError("Received an empty camera frame")

        output = frame.copy()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self._get_mesh().process(rgb)

        now = time.monotonic()

        # Face not detected
        if not results.multi_face_landmarks:

            with self._lock:

                self._closed_since = None
                self._eye_was_closed = False

                self._latest = {
                    **self._empty_status(),
                    "camera_available": True,
                    "message": (
                        "No face detected. "
                        "Move into the camera view."
                    )
                }

                self._last_frame = output

            cv2.putText(
                output,
                "No face detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

            return output

        landmarks = results.multi_face_landmarks[0].landmark

        height, width = frame.shape[:2]

        def xy(index):

            point = landmarks[index]

            return np.array(
                [
                    point.x * width,
                    point.y * height
                ],
                dtype=np.float64
            )

        # Eye landmarks
        left = np.array([xy(i) for i in LEFT_EYE])
        right = np.array([xy(i) for i in RIGHT_EYE])

        left_ear = eye_aspect_ratio(left)
        right_ear = eye_aspect_ratio(right)

        ear = (left_ear + right_ear) / 2.0

        # Gaze and head detection
        gaze = self._estimate_gaze(landmarks)
        head = self._estimate_head(landmarks)

        closed = ear < self.ear_threshold

        with self._lock:

            # Eye closure tracking
            if closed and self._closed_since is None:
                self._closed_since = now

            closed_seconds = (
                now - self._closed_since
                if closed and self._closed_since is not None
                else 0.0
            )

            # Blink counting
            if closed:

                self._eye_was_closed = True

            elif self._eye_was_closed:

                if (
                    self._closed_since is not None
                    and now - self._closed_since < 0.8
                ):
                    self._blink_count += 1

                self._eye_was_closed = False
                self._closed_since = None

            # Smooth detection history
            self._gaze_history.append(gaze)
            self._head_history.append(head)

            smooth_gaze = self._smooth(
                list(self._gaze_history),
                gaze
            )

            smooth_head = self._smooth(
                list(self._head_history),
                head
            )

            # Attention classification
            status = classify_attention(
                True,
                ear,
                closed_seconds,
                smooth_gaze,
                smooth_head,
                self.ear_threshold,
                self.drowsy_seconds
            )

            # NEW DYNAMIC ATTENTION SCORE
            score = calculate_attention_score(
                gaze=smooth_gaze,
                head_pose=smooth_head,
                ear=ear,
                closed_seconds=closed_seconds,
                ear_threshold=self.ear_threshold,
                drowsy_seconds=self.drowsy_seconds
            )

            # Short eye closure is not automatically drowsiness
            if closed and closed_seconds < self.drowsy_seconds:

                status = (
                    "Distracted"
                    if smooth_gaze in ("Left", "Right")
                    or smooth_head in (
                        "Turned left",
                        "Turned right",
                        "Lowered"
                    )
                    else "Focused"
                )

            self._latest = {
                "status": status,
                "attention_score": score,
                "ear": round(float(ear), 3),
                "blink_count": self._blink_count,
                "gaze": smooth_gaze,
                "head_pose": smooth_head,
                "eyes_closed_seconds": round(
                    closed_seconds, 2
                ),
                "alert": status == "Drowsy",
                "face_detected": True,
                "camera_available": True,
                "message": ""
            }

        # Draw eye landmarks
        for eye in (left, right):

            for point in eye.astype(int):

                cv2.circle(
                    output,
                    tuple(point),
                    2,
                    (150, 90, 255),
                    -1
                )

        # Face contour sample
        for index in (1, 10, 152, 234, 454):

            point = xy(index).astype(int)

            cv2.circle(
                output,
                tuple(point),
                2,
                (80, 210, 255),
                -1
            )

        # Display information
        cv2.putText(
            output,
            f"{status} | EAR: {ear:.2f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 230, 150),
            2
        )

        cv2.putText(
            output,
            f"Attention Score: {score}%",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if score >= 80 else (0, 165, 255),
            2
        )

        cv2.putText(
            output,
            f"Gaze: {smooth_gaze} | Head: {smooth_head}",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        with self._lock:
            self._last_frame = output

        return output

    def _estimate_gaze(self, landmarks):

        if len(landmarks) < 478:
            return "Uncertain"

        try:

            iris = landmarks[468]

            left_corner = landmarks[33]
            right_corner = landmarks[133]

            x1 = left_corner.x
            x2 = right_corner.x

            ratio = (
                (iris.x - min(x1, x2))
                / max(abs(x2 - x1), 1e-6)
            )

            if ratio < 0.38:
                return "Left"

            if ratio > 0.62:
                return "Right"

            return "Center"

        except (IndexError, AttributeError):

            return "Uncertain"

    @staticmethod
    def _estimate_head(landmarks):

        try:

            nose = landmarks[1]

            left_cheek = landmarks[234]
            right_cheek = landmarks[454]

            face_width = max(
                abs(right_cheek.x - left_cheek.x),
                1e-6
            )

            center = (
                left_cheek.x + right_cheek.x
            ) / 2

            horizontal_offset = (
                nose.x - center
            ) / face_width

            if horizontal_offset < -0.08:
                return "Turned left"

            if horizontal_offset > 0.08:
                return "Turned right"

            if nose.y > landmarks[152].y - 0.18:
                return "Lowered"

            return "Forward"

        except (IndexError, AttributeError):

            return "Uncertain"

    def latest(self):

        with self._lock:
            return dict(self._latest)

    def frame(self):

        with self._lock:

            return (
                None
                if self._last_frame is None
                else self._last_frame.copy()
            )

    def set_camera_message(self, message: str):

        with self._lock:

            self._latest = {
                **self._empty_status(),
                "camera_available": False,
                "message": message
            }

    def reset(self):

        with self._lock:

            self._closed_since = None
            self._eye_was_closed = False

            self._blink_count = 0

            self._gaze_history.clear()
            self._head_history.clear()

            self._latest = self._empty_status()
            self._last_frame = None


class CameraWorker:

    def __init__(
        self,
        detector: AttentionDetector,
        camera_index: int = 0
    ):

        self.detector = detector
        self.camera_index = camera_index

        self._capture = None
        self._thread = None

        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):

        with self._lock:

            if self._thread and self._thread.is_alive():
                return

            cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():

                cap.release()

                self.detector.set_camera_message(
                    "Camera unavailable. Check permission, "
                    "close other camera apps, and try again."
                )

                raise RuntimeError(
                    "Could not open webcam. "
                    "Check camera permission and connection."
                )

            self._capture = cap

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run,
                daemon=True
            )

            self._thread.start()

    def _run(self):

        while not self._stop_event.is_set():

            cap = self._capture

            if cap is None:
                break

            ok, frame = cap.read()

            if not ok or frame is None:

                self.detector.set_camera_message(
                    "Camera frame unavailable. Check your webcam."
                )

                time.sleep(0.1)
                continue

            try:

                self.detector.process(frame)

            except Exception as exc:

                self.detector.set_camera_message(
                    f"Detection error: {exc}"
                )

                time.sleep(0.1)

    def stop(self):

        self._stop_event.set()

        thread = self._thread

        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        with self._lock:

            if self._capture is not None:
                self._capture.release()

            self._capture = None
            self._thread = None

    def mjpeg_stream(self):

        while True:

            frame = self.detector.frame()

            if frame is None:

                frame = np.zeros(
                    (480, 640, 3),
                    dtype=np.uint8
                )

                cv2.putText(
                    frame,
                    "Start session to activate camera",
                    (35, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (220, 220, 220),
                    2
                )

            ok, encoded = cv2.imencode(".jpg", frame)

            if ok:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

            time.sleep(0.04)
