"""
🧠 Vision Input Layer — Face & Eye Extraction
Computes Eye Aspect Ratio (EAR) for blink detection and cognitive fatigue.

EAR formula (Soukupová & Čech, 2016):
    EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 × ||p1-p4|| )

    p1 = outer corner   p4 = inner corner
    p2,p3 = upper lid   p5,p6 = lower lid
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    FACE_MAX_FACES, FACE_REFINE_LANDMARKS,
    FACE_MIN_DETECTION_CONF, FACE_MIN_TRACKING_CONF,
    EAR_BLINK_THRESHOLD, EAR_DROWSY_THRESHOLD,
    BLINK_CONSEC_FRAMES, BLINK_RATE_WINDOW_SEC,
    NORMAL_BLINK_RATE, HIGH_BLINK_RATE, LOW_BLINK_RATE,
    RIGHT_EYE_EAR_PTS, LEFT_EYE_EAR_PTS,
)


@dataclass
class FaceFrame:
    """Cognitive-fatigue signals extracted from one frame."""
    timestamp:       float
    ear:             float           # mean EAR (both eyes)
    left_ear:        float
    right_ear:       float
    is_blinking:     bool
    blink_count:     int             # total blinks this session
    blink_rate_pm:   float           # blinks per minute (rolling 60 s window)
    drowsy_score:    float           # 0-1 continuous drowsiness estimate
    is_valid:        bool = True


class FaceExtractor:
    """
    Extracts real-time eye metrics for cognitive fatigue estimation.

    Usage:
        fe = FaceExtractor()
        frame = fe.process_frame(bgr_image)
        fe.release()
    """

    def __init__(self):
        self._mp_face = mp.solutions.face_mesh
        self.face_mesh = self._mp_face.FaceMesh(
            max_num_faces=FACE_MAX_FACES,
            refine_landmarks=FACE_REFINE_LANDMARKS,
            min_detection_confidence=FACE_MIN_DETECTION_CONF,
            min_tracking_confidence=FACE_MIN_TRACKING_CONF,
        )

        # Blink state machine
        self._consec_below: int = 0          # frames with EAR below threshold
        self._total_blinks: int = 0
        self._blink_times: Deque[float] = deque()   # timestamps of recent blinks

        # Drowsiness EMA
        self._drowsy_ema: float = 0.0
        self._ear_history: Deque[float] = deque(maxlen=90)  # ~3 s at 30 FPS

    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray) -> Optional[FaceFrame]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceFrame(
                timestamp=time.time(), ear=0.3, left_ear=0.3, right_ear=0.3,
                is_blinking=False, blink_count=self._total_blinks,
                blink_rate_pm=self._blink_rate(), drowsy_score=self._drowsy_ema,
                is_valid=False,
            )

        h, w = bgr_frame.shape[:2]
        face_lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_lm], dtype=np.float32)

        left_ear  = self._ear(pts, LEFT_EYE_EAR_PTS)
        right_ear = self._ear(pts, RIGHT_EYE_EAR_PTS)
        mean_ear  = (left_ear + right_ear) / 2.0

        self._ear_history.append(mean_ear)
        is_blinking = self._update_blink_state(mean_ear)
        drowsy = self._update_drowsiness(mean_ear)
        ts = time.time()

        return FaceFrame(
            timestamp=ts,
            ear=mean_ear,
            left_ear=left_ear,
            right_ear=right_ear,
            is_blinking=is_blinking,
            blink_count=self._total_blinks,
            blink_rate_pm=self._blink_rate(),
            drowsy_score=drowsy,
        )

    def draw_eye_landmarks(self, bgr_frame: np.ndarray,
                           face_frame: Optional[FaceFrame],
                           ear: float) -> np.ndarray:
        """Overlay EAR value and eye state indicator."""
        out = bgr_frame.copy()
        if face_frame is None:
            return out
        colour = (0, 255, 0) if ear > EAR_DROWSY_THRESHOLD else (0, 60, 255)
        label = f"EAR: {ear:.3f}"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2, cv2.LINE_AA)
        if face_frame.is_blinking:
            cv2.putText(out, "BLINK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 160, 0), 2, cv2.LINE_AA)
        return out

    def release(self):
        self.face_mesh.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ear(pts: np.ndarray, indices: list) -> float:
        """EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)"""
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in indices]
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = 2.0 * np.linalg.norm(p1 - p4) + 1e-8
        return float(vertical / horizontal)

    def _update_blink_state(self, ear: float) -> bool:
        """Simple state machine: blink event when EAR dips below threshold."""
        is_blinking = False
        if ear < EAR_BLINK_THRESHOLD:
            self._consec_below += 1
        else:
            if self._consec_below >= BLINK_CONSEC_FRAMES:
                self._total_blinks += 1
                self._blink_times.append(time.time())
                is_blinking = True
            self._consec_below = 0
        return is_blinking

    def _update_drowsiness(self, ear: float) -> float:
        """
        Drowsiness score [0,1]:
        - Weighted combination of mean EAR depression and sustained low-EAR events.
        - Smoothed with an exponential moving average.
        """
        if len(self._ear_history) < 10:
            return 0.0

        mean_ear = float(np.mean(self._ear_history))
        # Invert: lower EAR → higher drowsiness
        ear_score = max(0.0, (EAR_DROWSY_THRESHOLD - mean_ear) / EAR_DROWSY_THRESHOLD)

        # Blink-rate component
        bpm = self._blink_rate()
        if bpm < LOW_BLINK_RATE:
            br_score = 0.4   # low blink rate → fatigue / focused dissociation
        elif bpm > HIGH_BLINK_RATE:
            br_score = 0.6   # high blink rate → eye strain
        else:
            br_score = 0.0

        raw = 0.7 * ear_score + 0.3 * br_score
        alpha = 0.05                             # slow-moving EMA
        self._drowsy_ema = alpha * raw + (1 - alpha) * self._drowsy_ema
        return float(np.clip(self._drowsy_ema, 0.0, 1.0))

    def _blink_rate(self) -> float:
        """Blinks per minute over the last BLINK_RATE_WINDOW_SEC seconds."""
        now = time.time()
        cutoff = now - BLINK_RATE_WINDOW_SEC
        # Prune old entries
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        elapsed = min(now - (self._blink_times[0] if self._blink_times else now),
                      BLINK_RATE_WINDOW_SEC)
        if elapsed < 5.0:
            return NORMAL_BLINK_RATE   # not enough data yet
        return len(self._blink_times) / (elapsed / 60.0)