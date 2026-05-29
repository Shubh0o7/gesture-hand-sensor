"""
Phase 1: Perception Layer - Hand Landmark Tracker
Uses MediaPipe Hands to detect 21 3D hand landmarks from webcam.
Extracts spatial (X, Y, Z) coordinates and normalizes them for
invariance to hand distance from camera.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


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


class HandTracker:
    """Real-time hand landmark tracker using MediaPipe Hands."""

    # 21 landmark names for reference
    LANDMARK_NAMES = [
        'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
    ]

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        detection_confidence: Optional[float] = None,
        tracking_confidence: Optional[float] = None,
    ):
        """
        Initialize the HandTracker.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
            detection_confidence: Alias for min_detection_confidence.
            tracking_confidence: Alias for min_tracking_confidence.
        """
        if detection_confidence is not None:
            min_detection_confidence = detection_confidence
        if tracking_confidence is not None:
            min_tracking_confidence = tracking_confidence

        self.processor = HandLandmarkProcessor()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._camera = None

    def start(self, camera_index: int = 0) -> bool:
        """Open the default webcam (optional helper for scripts)."""
        self._camera = cv2.VideoCapture(camera_index)
        return self._camera.isOpened()

    def get_landmarks(self) -> Optional[np.ndarray]:
        """Read one frame from the opened camera and return normalized landmarks."""
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
        Process a single video frame and extract hand landmarks.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Tuple of (normalized_landmarks_array or None, annotated_frame)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process the frame
        results = self.hands.process(rgb_frame)

        # Draw landmarks on the frame
        annotated_frame = frame.copy()
        landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            # Extract landmarks from the first detected hand
            raw_landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
            landmarks = self._normalize_landmarks(raw_landmarks)

        return landmarks, annotated_frame

    def process(self, frame: np.ndarray) -> dict:
        """
        Process a frame and return a dict for pipeline integration.

        Returns:
            {"landmarks": [np.ndarray], "annotated_frame": np.ndarray}
        """
        landmarks, annotated = self.process_frame(frame)
        return {
            "landmarks": [landmarks] if landmarks is not None else [],
            "annotated_frame": annotated,
        }

    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Extract raw (X, Y, Z) coordinates from MediaPipe hand landmarks.

        Args:
            hand_landmarks: MediaPipe hand landmarks object.

        Returns:
            numpy array of shape (21, 3) with raw coordinates.
        """
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Delegate to HandLandmarkProcessor for consistent normalization."""
        return self.processor.normalize(landmarks)

    def get_landmark_vector(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Flatten the 21x3 landmark array into a 63-element feature vector.

        Args:
            landmarks: Normalized landmarks of shape (21, 3).

        Returns:
            Flattened array of shape (63,).
        """
        return landmarks.flatten()

    def release(self):
        """Release MediaPipe and camera resources."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self.hands.close()

    def close(self):
        """Alias for release()."""
        self.release()


def run_live_demo():
    """Run a live webcam demo showing hand landmark detection."""
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Hand Tracker Live Demo - Press 'q' to quit")
    print("Show your hand to the camera to see 21 landmarks detected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        landmarks, annotated_frame = tracker.process_frame(frame)

        if landmarks is not None:
            vector = tracker.get_landmark_vector(landmarks)
            # Display landmark count and vector norm
            info_text = f"Landmarks: 21 | Vector dim: {len(vector)} | Norm: {np.linalg.norm(vector):.2f}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Hand Tracker - Phase 1', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


if __name__ == '__main__':
    run_live_demo()
