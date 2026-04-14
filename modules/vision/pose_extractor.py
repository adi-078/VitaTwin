"""
🎥 Vision Input Layer — Pose Extraction
Captures joint coordinates, body posture, and movement trajectory
using MediaPipe Pose (33 landmarks).
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    JOINT_INDICES,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
)


@dataclass
class PoseFrame:
    """One frame of extracted pose data."""
    timestamp:           float
    landmarks:           np.ndarray          # shape (33, 3)  — normalized x,y,z
    landmark_visibility: np.ndarray          # shape (33,)
    joint_angles:        Dict[str, float]    # degrees
    joint_velocities:    Dict[str, float]    # degrees/s
    movement_speed:      float               # mean wrist/knee speed (norm-coords/s)
    joint_stability:     float               # std of key-joint displacements
    is_valid:            bool = True


class PoseExtractor:
    """
    Real-time pose extractor.

    Usage:
        extractor = PoseExtractor()
        frame = extractor.process_frame(bgr_image)
        annotated = extractor.draw_skeleton(bgr_image, frame)
        extractor.release()
    """

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self.pose = self._mp_pose.Pose(
            model_complexity=POSE_MODEL_COMPLEXITY,
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
            smooth_landmarks=True,
        )
        self._prev_frame: Optional[PoseFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray) -> Optional[PoseFrame]:
        """
        Process a BGR camera frame.
        Returns PoseFrame on success, None if pose not detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )
        visibility = np.array(
            [lm.visibility for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )
        ts = time.time()
        angles    = self._compute_joint_angles(landmarks)
        velocities = self._compute_velocities(angles, ts)
        speed     = self._compute_movement_speed(landmarks, ts)
        stability = self._compute_joint_stability(landmarks)

        frame = PoseFrame(
            timestamp=ts,
            landmarks=landmarks,
            landmark_visibility=visibility,
            joint_angles=angles,
            joint_velocities=velocities,
            movement_speed=speed,
            joint_stability=stability,
        )
        self._prev_frame = frame
        return frame

    def draw_skeleton(self, bgr_frame: np.ndarray,
                      pose_frame: Optional[PoseFrame]) -> np.ndarray:
        """Draw colour-coded skeleton overlay."""
        if pose_frame is None:
            return bgr_frame

        out = bgr_frame.copy()
        h, w = bgr_frame.shape[:2]
        lm  = pose_frame.landmarks
        vis = pose_frame.landmark_visibility

        CONNECTIONS = [
            ("left_shoulder",  "left_elbow"),
            ("left_elbow",     "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow",    "right_wrist"),
            ("left_shoulder",  "right_shoulder"),
            ("left_shoulder",  "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip",       "right_hip"),
            ("left_hip",       "left_knee"),
            ("left_knee",      "left_ankle"),
            ("right_hip",      "right_knee"),
            ("right_knee",     "right_ankle"),
        ]
        for j1, j2 in CONNECTIONS:
            i1, i2 = JOINT_INDICES[j1], JOINT_INDICES[j2]
            if vis[i1] > 0.5 and vis[i2] > 0.5:
                p1 = (int(lm[i1, 0] * w), int(lm[i1, 1] * h))
                p2 = (int(lm[i2, 0] * w), int(lm[i2, 1] * h))
                cv2.line(out, p1, p2, (0, 255, 180), 2, cv2.LINE_AA)

        for name, idx in JOINT_INDICES.items():
            if vis[idx] > 0.5:
                x = int(lm[idx, 0] * w)
                y = int(lm[idx, 1] * h)
                cv2.circle(out, (x, y), 5, (255, 200, 0), -1, cv2.LINE_AA)

        return out

    def release(self):
        self.pose.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at vertex b, given three 2-D points a, b, c (degrees)."""
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        cos_a = np.dot(ba, bc) / denom
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _compute_joint_angles(self, lm: np.ndarray) -> Dict[str, float]:
        ji = JOINT_INDICES
        a  = self._angle_3pts  # shorthand

        return {
            "left_knee":      a(lm[ji["left_hip"],  :2], lm[ji["left_knee"],  :2], lm[ji["left_ankle"],  :2]),
            "right_knee":     a(lm[ji["right_hip"], :2], lm[ji["right_knee"], :2], lm[ji["right_ankle"], :2]),
            "left_hip":       a(lm[ji["left_shoulder"],  :2], lm[ji["left_hip"],  :2], lm[ji["left_knee"],  :2]),
            "right_hip":      a(lm[ji["right_shoulder"], :2], lm[ji["right_hip"], :2], lm[ji["right_knee"], :2]),
            "left_elbow":     a(lm[ji["left_shoulder"],  :2], lm[ji["left_elbow"],  :2], lm[ji["left_wrist"],  :2]),
            "right_elbow":    a(lm[ji["right_shoulder"], :2], lm[ji["right_elbow"], :2], lm[ji["right_wrist"], :2]),
            "left_shoulder":  a(lm[ji["left_hip"],  :2], lm[ji["left_shoulder"],  :2], lm[ji["left_elbow"],  :2]),
            "right_shoulder": a(lm[ji["right_hip"], :2], lm[ji["right_shoulder"], :2], lm[ji["right_elbow"], :2]),
            "trunk_lean":     self._trunk_lean(lm),
        }

    @staticmethod
    def _trunk_lean(lm: np.ndarray) -> float:
        """Forward trunk lean angle from vertical (degrees)."""
        ji = JOINT_INDICES
        mid_sh  = (lm[ji["left_shoulder"]] + lm[ji["right_shoulder"]]) / 2
        mid_hip = (lm[ji["left_hip"]]      + lm[ji["right_hip"]]) / 2
        v = mid_sh[:2] - mid_hip[:2]
        up = np.array([0.0, -1.0])
        cos_a = np.dot(v, up) / (np.linalg.norm(v) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(abs(cos_a), 0.0, 1.0))))

    def _compute_velocities(self, curr: Dict[str, float],
                            ts: float) -> Dict[str, float]:
        if self._prev_frame is None:
            return {k: 0.0 for k in curr}
        dt = ts - self._prev_frame.timestamp
        if dt < 1e-6:
            return {k: 0.0 for k in curr}
        prev = self._prev_frame.joint_angles
        return {k: (v - prev.get(k, v)) / dt for k, v in curr.items()}

    def _compute_movement_speed(self, lm: np.ndarray, ts: float) -> float:
        """Mean velocity of wrists and knees (normalized coords / s)."""
        if self._prev_frame is None:
            return 0.0
        dt = ts - self._prev_frame.timestamp
        if dt < 1e-6:
            return 0.0
        key = [
            JOINT_INDICES["left_wrist"],  JOINT_INDICES["right_wrist"],
            JOINT_INDICES["left_knee"],   JOINT_INDICES["right_knee"],
        ]
        dists = [np.linalg.norm(lm[i, :2] - self._prev_frame.landmarks[i, :2]) for i in key]
        return float(np.mean(dists)) / dt

    def _compute_joint_stability(self, lm: np.ndarray) -> float:
        """
        Stability metric = std of displacements across key lower-body joints.
        Higher value means more erratic / unstable motion.
        """
        if self._prev_frame is None:
            return 0.0
        key = [
            JOINT_INDICES["left_knee"],   JOINT_INDICES["right_knee"],
            JOINT_INDICES["left_hip"],    JOINT_INDICES["right_hip"],
            JOINT_INDICES["left_ankle"],  JOINT_INDICES["right_ankle"],
        ]
        dists = [np.linalg.norm(lm[i, :2] - self._prev_frame.landmarks[i, :2]) for i in key]
        return float(np.std(dists))