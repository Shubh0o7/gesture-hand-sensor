"""
Phase 1: Perception Layer - Hand Landmark Tracker
Uses MediaPipe Hand Landmarker (Tasks API) for real-time 21-point hand tracking.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def ensure_hand_model() -> Path:
    """Download the Hand Landmarker model if it is not present locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1000:
        return MODEL_PATH

    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    tmp_path = MODEL_PATH.with_suffix(".task.download")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(MODEL_PATH)
    print("Model download complete.")
    return MODEL_PATH


class HandLandmarkProcessor:
    """Utilities for normalizing and flattening hand landmark sequences."""

    NUM_LANDMARKS = 21
    DIMS_PER_LANDMARK = 3

    @staticmethod
    def normalize(landmarks: np.ndarray) -> np.ndarray:
        """Wrist-centered, scale-invariant normalization."""
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        reference_distance = np.linalg.norm(centered[9])
        if reference_distance > 1e-6:
            return centered / reference_distance
        return centered

    @staticmethod
    def flatten(landmarks: np.ndarray) -> np.ndarray:
        """Flatten (21, 3) landmarks to a 63-d feature vector."""
        return landmarks.flatten().astype(np.float32)

    @staticmethod
    def sequence_to_features(sequence: list) -> np.ndarray:
        """Convert a list of landmark frames to (T, 63) array."""
        rows = []
        for frame in sequence:
            arr = np.asarray(frame, dtype=np.float32)
            if arr.shape == (21, 3):
                rows.append(HandLandmarkProcessor.flatten(arr))
            else:
                rows.append(arr.reshape(-1))
        return np.stack(rows, axis=0)


def _draw_hand_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    """Draw hand skeleton on a BGR frame using OpenCV."""
    height, width = frame.shape[:2]
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 255), 2)


class HandTracker:
    """Real-time hand landmark tracker using MediaPipe Hand Landmarker."""

    LANDMARK_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        detection_confidence: Optional[float] = None,
        tracking_confidence: Optional[float] = None,
        model_path: Optional[str] = None,
    ):
        if detection_confidence is not None:
            min_detection_confidence = detection_confidence
        if tracking_confidence is not None:
            min_tracking_confidence = tracking_confidence

        self.processor = HandLandmarkProcessor()
        self.max_hands = max_hands
        self._camera = None
        self._frame_timestamp_ms = 0

        resolved_model = Path(model_path) if model_path else ensure_hand_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=str(resolved_model))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def start(self, camera_index: int = 0) -> bool:
        """Open the default webcam."""
        self._camera = cv2.VideoCapture(camera_index)
        return self._camera.isOpened()

    def get_landmarks(self) -> Optional[np.ndarray]:
        """Read one frame and return normalized landmarks for the first hand."""
        if self._camera is None or not self._camera.isOpened():
            return None
        ok, frame = self._camera.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        landmarks, _ = self.process_frame(frame)
        return landmarks

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Process a BGR frame and extract normalized landmarks.

        Returns:
            (landmarks array shape (21, 3) or None, annotated BGR frame)
        """
        annotated_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33
        result = self.detector.detect_for_video(mp_image, self._frame_timestamp_ms)

        landmarks = None
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                _draw_hand_landmarks(annotated_frame, hand_landmarks)

            raw = self._extract_landmarks(result.hand_landmarks[0])
            landmarks = self._normalize_landmarks(raw)

        return landmarks, annotated_frame

    def process(self, frame: np.ndarray) -> dict:
        """Pipeline-friendly wrapper returning landmarks and annotated frame."""
        landmarks, annotated = self.process_frame(frame)
        return {
            "landmarks": [landmarks] if landmarks is not None else [],
            "annotated_frame": annotated,
        }

    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        return np.array(coords, dtype=np.float32)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        return self.processor.normalize(landmarks)

    def get_landmark_vector(self, landmarks: np.ndarray) -> np.ndarray:
        return landmarks.flatten()

    def release(self):
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None

    def close(self):
        self.release()


def run_live_demo():
    """Live webcam demo for the hand sensor."""
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Hand Tracker - press 'q' to quit")
    print("Show your hand to the camera.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame.")
                break

            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame = tracker.process_frame(frame)

            if landmarks is not None:
                vector = tracker.get_landmark_vector(landmarks)
                info = (
                    f"Landmarks: 21 | Vector: {len(vector)} | "
                    f"Norm: {np.linalg.norm(vector):.2f}"
                )
                cv2.putText(
                    annotated_frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    annotated_frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imshow("Hand Tracker", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()


if __name__ == "__main__":
    run_live_demo()
