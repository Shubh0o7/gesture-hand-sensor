"""
Phase 1: Data Acquisition - Gesture Data Collector
Uses OpenCV to record video sequences of dynamic gestures and saves
the MediaPipe hand landmark coordinates into structured CSV datasets.
This avoids saving raw images, saving storage and speeding up training.
"""

import cv2
import csv
import json
import os
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from hand_tracker import HandTracker


class GestureDataCollector:
    """Collects gesture landmark data from webcam into CSV/JSON datasets."""

    # Predefined gesture classes for the robot control system
    GESTURE_CLASSES = [
        'forward',    # Push gesture - move forward
        'backward',   # Pull gesture - move backward
        'turn_left',  # Swipe left - turn left
        'turn_right', # Swipe right - turn right
        'stop',       # Open palm - stop
        'slow',       # Closing fist slowly - reduce speed
        'fast',       # Opening fist quickly - increase speed
        'idle'        # No meaningful gesture / resting
    ]

    def __init__(self, data_dir: str = '../data', sequence_length: int = 45,
                 fps: int = 30):
        """
        Initialize the GestureDataCollector.

        Args:
            data_dir: Directory to save collected data.
            sequence_length: Number of frames per gesture sequence (30-60).
            fps: Target frames per second for recording.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.sequence_length = sequence_length
        self.fps = fps
        self.tracker = HandTracker(max_hands=1)

        # Create data directory structure
        os.makedirs(self.data_dir, exist_ok=True)
        for gesture in self.GESTURE_CLASSES:
            os.makedirs(os.path.join(self.data_dir, gesture), exist_ok=True)

    def collect_gesture_sequence(self, gesture_label: str) -> Optional[List[np.ndarray]]:
        """
        Record a single gesture sequence from webcam.

        Args:
            gesture_label: The gesture class being recorded.

        Returns:
            List of normalized landmark arrays (each shape 21x3), or None if failed.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam.")
            return None

        sequence = []
        recording = False
        countdown = 3  # 3-second countdown before recording

        print(f"\nPrepare to perform: '{gesture_label}'")
        print(f"Recording {self.sequence_length} frames (~{self.sequence_length/self.fps:.1f}s)")
        print("Press SPACE to start countdown, 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame = self.tracker.process_frame(frame)

            if not recording:
                # Show instructions
                cv2.putText(annotated_frame, f"Gesture: {gesture_label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(annotated_frame, "Press SPACE to start",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                cv2.imshow('Data Collector', annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    # Countdown
                    for i in range(countdown, 0, -1):
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.flip(frame, 1)
                            cv2.putText(frame, str(i), (frame.shape[1]//2 - 30, frame.shape[0]//2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                            cv2.imshow('Data Collector', frame)
                            cv2.waitKey(1000)
                    recording = True
                    print("Recording...")

                elif key == ord('q'):
                    break
            else:
                # Recording mode
                if landmarks is not None:
                    sequence.append(landmarks)

                progress = len(sequence)
                total = self.sequence_length
                bar_width = 300
                filled = int(bar_width * progress / total)

                # Progress bar
                cv2.rectangle(annotated_frame, (10, 50), (10 + bar_width, 70), (50, 50, 50), -1)
                cv2.rectangle(annotated_frame, (10, 50), (10 + filled, 70), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"Recording: {progress}/{total}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Data Collector', annotated_frame)
                cv2.waitKey(max(1, int(1000 / self.fps)))

                if len(sequence) >= self.sequence_length:
                    print(f"Recorded {len(sequence)} frames.")
                    break

        cap.release()
        cv2.destroyAllWindows()

        if len(sequence) >= self.sequence_length:
            return sequence[:self.sequence_length]
        return None

    def save_sequence_csv(self, sequence: List[np.ndarray], gesture_label: str,
                          sample_id: Optional[str] = None) -> str:
        """
        Save a gesture sequence to a CSV file.

        CSV format: Each row is one frame with columns:
        frame_idx, lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ..., lm20_x, lm20_y, lm20_z

        Args:
            sequence: List of landmark arrays.
            gesture_label: Gesture class name.
            sample_id: Optional unique identifier for this sample.

        Returns:
            Path to the saved CSV file.
        """
        if sample_id is None:
            sample_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        filename = f"{gesture_label}_{sample_id}.csv"
        filepath = os.path.join(self.data_dir, gesture_label, filename)

        # Build header
        header = ['frame_idx']
        for i in range(21):
            header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for frame_idx, landmarks in enumerate(sequence):
                row = [frame_idx] + landmarks.flatten().tolist()
                writer.writerow(row)

        print(f"Saved: {filepath}")
        return filepath

    def save_sequence_json(self, sequence: List[np.ndarray], gesture_label: str,
                           sample_id: Optional[str] = None) -> str:
        """
        Save a gesture sequence to a JSON file.

        JSON format:
        {
            "gesture": "forward",
            "sequence_length": 45,
            "timestamp": "...",
            "frames": [
                {"frame_idx": 0, "landmarks": [[x,y,z], [x,y,z], ...]},
                ...
            ]
        }

        Args:
            sequence: List of landmark arrays.
            gesture_label: Gesture class name.
            sample_id: Optional unique identifier.

        Returns:
            Path to the saved JSON file.
        """
        if sample_id is None:
            sample_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        filename = f"{gesture_label}_{sample_id}.json"
        filepath = os.path.join(self.data_dir, gesture_label, filename)

        data = {
            'gesture': gesture_label,
            'sequence_length': len(sequence),
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'frames': []
        }

        for frame_idx, landmarks in enumerate(sequence):
            data['frames'].append({
                'frame_idx': frame_idx,
                'landmarks': landmarks.tolist()
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved: {filepath}")
        return filepath

    def generate_synthetic_data(self, samples_per_class: int = 50) -> Dict[str, int]:
        """
        Generate synthetic training data by creating randomized landmark sequences.
        Useful for initial model development before collecting real data.

        Each gesture class gets distinct motion patterns:
        - forward: landmarks shift upward over time
        - backward: landmarks shift downward
        - turn_left: landmarks shift left
        - turn_right: landmarks shift right
        - stop: landmarks remain relatively static (open palm)
        - slow: landmarks converge (closing fist)
        - fast: landmarks diverge (opening fist)
        - idle: random small movements

        Args:
            samples_per_class: Number of synthetic samples per gesture class.

        Returns:
            Dictionary mapping gesture labels to number of samples created.
        """
        counts = {}
        np.random.seed(42)

        for gesture in self.GESTURE_CLASSES:
            for sample_idx in range(samples_per_class):
                sequence = self._generate_synthetic_sequence(gesture)
                sample_id = f"synthetic_{sample_idx:04d}"
                self.save_sequence_csv(sequence, gesture, sample_id)
                
            counts[gesture] = samples_per_class
            print(f"Generated {samples_per_class} synthetic samples for '{gesture}'")

        return counts

    def _generate_synthetic_sequence(self, gesture: str) -> List[np.ndarray]:
        """
        Generate a single synthetic gesture sequence.

        Args:
            gesture: Gesture class name.

        Returns:
            List of landmark arrays simulating the gesture.
        """
        sequence = []

        # Base hand pose (normalized, centered at origin)
        base_pose = np.random.randn(21, 3).astype(np.float32) * 0.3
        base_pose[0] = [0, 0, 0]  # Wrist at origin

        for frame_idx in range(self.sequence_length):
            t = frame_idx / self.sequence_length  # Normalized time [0, 1]
            noise = np.random.randn(21, 3).astype(np.float32) * 0.02

            if gesture == 'forward':
                # Landmarks shift in -Y direction (push forward)
                delta = np.zeros((21, 3), dtype=np.float32)
                delta[:, 1] = -t * 0.5
                delta[:, 2] = -t * 0.3  # Also moves toward camera
            elif gesture == 'backward':
                delta = np.zeros((21, 3), dtype=np.float32)
                delta[:, 1] = t * 0.5
                delta[:, 2] = t * 0.3
            elif gesture == 'turn_left':
                delta = np.zeros((21, 3), dtype=np.float32)
                delta[:, 0] = -t * 0.6  # Shift left
            elif gesture == 'turn_right':
                delta = np.zeros((21, 3), dtype=np.float32)
                delta[:, 0] = t * 0.6  # Shift right
            elif gesture == 'stop':
                # Open palm - fingers spread, minimal movement
                delta = np.zeros((21, 3), dtype=np.float32)
                delta[4:, :] = np.sin(t * np.pi) * 0.05  # Slight spread
            elif gesture == 'slow':
                # Closing fist - fingertips converge toward palm
                delta = np.zeros((21, 3), dtype=np.float32)
                for tip_idx in [4, 8, 12, 16, 20]:  # Fingertips
                    delta[tip_idx] = -base_pose[tip_idx] * t * 0.5
            elif gesture == 'fast':
                # Opening fist - fingertips diverge from palm
                delta = np.zeros((21, 3), dtype=np.float32)
                for tip_idx in [4, 8, 12, 16, 20]:
                    delta[tip_idx] = base_pose[tip_idx] * t * 0.5
            else:  # idle
                delta = np.random.randn(21, 3).astype(np.float32) * 0.03

            frame_landmarks = base_pose + delta + noise
            # Re-normalize: center on wrist
            frame_landmarks -= frame_landmarks[0]
            ref_dist = np.linalg.norm(frame_landmarks[9])
            if ref_dist > 1e-6:
                frame_landmarks /= ref_dist

            sequence.append(frame_landmarks)

        return sequence

    def get_dataset_stats(self) -> Dict[str, int]:
        """
        Get statistics about the collected dataset.

        Returns:
            Dictionary mapping gesture labels to sample counts.
        """
        stats = {}
        for gesture in self.GESTURE_CLASSES:
            gesture_dir = os.path.join(self.data_dir, gesture)
            if os.path.exists(gesture_dir):
                csv_files = [f for f in os.listdir(gesture_dir) if f.endswith('.csv')]
                json_files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
                stats[gesture] = len(csv_files) + len(json_files)
            else:
                stats[gesture] = 0
        return stats

    def release(self):
        """Release resources."""
        self.tracker.release()


def main():
    """Interactive data collection session."""
    collector = GestureDataCollector(data_dir='../data', sequence_length=45)

    print("="*60)
    print("  Gesture Data Collector - Phase 1")
    print("="*60)
    print("\nAvailable gestures:")
    for i, gesture in enumerate(GestureDataCollector.GESTURE_CLASSES):
        print(f"  {i}: {gesture}")

    print("\nCommands:")
    print("  [0-7]  - Record a gesture")
    print("  's'    - Generate synthetic data")
    print("  'i'    - Show dataset info")
    print("  'q'    - Quit")

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd == 'q':
            break
        elif cmd == 's':
            print("Generating synthetic training data...")
            counts = collector.generate_synthetic_data(samples_per_class=50)
            print(f"\nGenerated: {sum(counts.values())} total samples")
        elif cmd == 'i':
            stats = collector.get_dataset_stats()
            print("\nDataset Statistics:")
            for gesture, count in stats.items():
                print(f"  {gesture}: {count} samples")
            print(f"  Total: {sum(stats.values())} samples")
        elif cmd.isdigit() and 0 <= int(cmd) < len(GestureDataCollector.GESTURE_CLASSES):
            gesture = GestureDataCollector.GESTURE_CLASSES[int(cmd)]
            sequence = collector.collect_gesture_sequence(gesture)
            if sequence:
                collector.save_sequence_csv(sequence, gesture)
                print(f"Successfully recorded '{gesture}' gesture!")
            else:
                print("Recording failed or was cancelled.")
        else:
            print("Invalid command. Try again.")

    collector.release()
    print("Data collection session ended.")


if __name__ == '__main__':
    main()
