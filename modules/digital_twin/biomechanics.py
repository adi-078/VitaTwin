"""
🤖 Digital Twin — Biomechanics Engine
Estimates internal joint loads (torques, mechanical stress) from
MediaPipe pose landmarks using a simplified inverse-dynamics model.

Instead of a full MuJoCo physics sim (which would add >100 ms latency),
we use segment-mass Newton-Euler inverse dynamics on the sagittal plane.
This gives physically meaningful torque estimates in real time.

Reference: Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    JOINT_INDICES,
    DEFAULT_BODY_MASS_KG,
    SEGMENT_MASS_FRACTIONS,
    SEGMENT_LENGTH_FRACTIONS,
    GRAVITY,
)
from modules.vision.pose_extractor import PoseFrame


@dataclass
class BiomechanicsFrame:
    """Estimated internal joint loads for one frame."""
    timestamp: float

    # Joint torques (N·m) — positive = extensor
    torques: Dict[str, float] = field(default_factory=dict)

    # Mechanical stress index [0-100] per joint
    # Normalised relative to max voluntary torque estimates
    stress_index: Dict[str, float] = field(default_factory=dict)

    # Cumulative mechanical load this session
    cumulative_load: float = 0.0

    # Overall biomechanical fatigue contribution [0-100]
    biomech_fatigue: float = 0.0


# Maximum voluntary torque estimates (N·m) for a 70 kg person
# Used for normalisation.  Scaled linearly with body mass.
MAX_VOLUNTARY_TORQUES: Dict[str, float] = {
    "left_knee":      165.0,
    "right_knee":     165.0,
    "left_hip":       200.0,
    "right_hip":      200.0,
    "left_ankle":      80.0,
    "right_ankle":     80.0,
    "left_shoulder":  100.0,
    "right_shoulder": 100.0,
}


class BiomechanicsEngine:
    """
    Computes joint torques from pose data using segment-mass inverse dynamics.

    The retargeting chain:
        MediaPipe 2-D landmarks (normalized)
         → pixel coordinates (scale by frame dims)
         → world coordinates (scale by estimated height)
         → segment COM positions
         → segment accelerations (finite diff × 2)
         → inverse dynamics  → joint torques

    Usage:
        engine = BiomechanicsEngine(body_mass_kg=75.0, height_m=1.78)
        bframe = engine.process(pose_frame, frame_dims=(1280, 720))
    """

    def __init__(self, body_mass_kg: float = DEFAULT_BODY_MASS_KG,
                 height_m: float = 1.75):
        self.mass   = body_mass_kg
        self.height = height_m
        self._prev1: Optional[PoseFrame] = None
        self._prev2: Optional[PoseFrame] = None
        self._cumulative_load: float = 0.0
        self._stress_history: deque = deque(maxlen=300)  # 1-min at 5 Hz

    def process(self, pose_frame: Optional[PoseFrame],
                frame_dims=(1280, 720)) -> Optional[BiomechanicsFrame]:
        """Compute biomechanics for one frame. Requires ≥3 consecutive frames."""
        if pose_frame is None:
            return None

        torques = {}
        stress  = {}

        if self._prev1 is not None and self._prev2 is not None:
            torques, stress = self._compute_torques(
                self._prev2, self._prev1, pose_frame, frame_dims
            )

        # ── Update history ────────────────────────────────────────────────────
        self._prev2 = self._prev1
        self._prev1 = pose_frame

        if torques:
            # Mechanical load contribution = RMS torque × dt
            dt = pose_frame.timestamp - (self._prev2.timestamp if self._prev2 else pose_frame.timestamp - 0.033)
            rms = float(np.sqrt(np.mean([t**2 for t in torques.values()])))
            self._cumulative_load += rms * dt
            self._stress_history.append(float(np.mean(list(stress.values()))))

        biomech_fat = self._biomech_fatigue_score()

        return BiomechanicsFrame(
            timestamp=pose_frame.timestamp,
            torques=torques,
            stress_index=stress,
            cumulative_load=self._cumulative_load,
            biomech_fatigue=biomech_fat,
        )

    def reset_session(self):
        self._cumulative_load = 0.0
        self._stress_history.clear()
        self._prev1 = self._prev2 = None

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _landmark_to_world(self, lm: np.ndarray, frame_dims) -> np.ndarray:
        """
        Convert normalized MediaPipe coords → approximate world coords (m).
        We use body height as the scale reference.
        """
        w, h = frame_dims
        pixel = lm[:, :2] * np.array([w, h])  # (33, 2) in pixels
        # Estimate pixel/metre from known body proportions
        ji = JOINT_INDICES
        head_y  = lm[ji["nose"], 1] * h
        ankle_y = (lm[ji["left_ankle"], 1] + lm[ji["right_ankle"], 1]) / 2 * h
        pixel_height = abs(ankle_y - head_y) + 1e-3
        scale = self.height / pixel_height   # metres per pixel
        world = pixel * scale
        return world  # (33, 2) in metres, +y = down

    @staticmethod
    def _com_of_segment(p_proximal: np.ndarray, p_distal: np.ndarray,
                        fraction: float = 0.5) -> np.ndarray:
        """Segment COM is at 'fraction' from proximal end."""
        return p_proximal + fraction * (p_distal - p_proximal)

    def _compute_torques(self, f0: PoseFrame, f1: PoseFrame, f2: PoseFrame,
                         frame_dims) -> tuple:
        """
        Finite-difference acceleration → Newton-Euler inverse dynamics.
        Returns (torques, stress_indices) dicts.
        """
        ji = JOINT_INDICES
        dt1 = f1.timestamp - f0.timestamp
        dt2 = f2.timestamp - f1.timestamp
        dt  = (dt1 + dt2) / 2.0
        if dt < 1e-4:
            return {}, {}

        w0 = self._landmark_to_world(f0.landmarks, frame_dims)
        w1 = self._landmark_to_world(f1.landmarks, frame_dims)
        w2 = self._landmark_to_world(f2.landmarks, frame_dims)

        # Second-order central difference: a ≈ (x2 - 2x1 + x0) / dt²
        acc = (w2 - 2 * w1 + w0) / (dt * dt)  # (33, 2)  m/s²

        torques = {}
        stress  = {}

        # ── Knee torques (most loaded joint in squats/walking) ───────────────
        for side in ("left", "right"):
            hip_i   = ji[f"{side}_hip"]
            knee_i  = ji[f"{side}_knee"]
            ankle_i = ji[f"{side}_ankle"]

            shank_mass = self.mass * SEGMENT_MASS_FRACTIONS["shank"]
            shank_len  = self.height * SEGMENT_LENGTH_FRACTIONS["shank"]
            shank_com  = self._com_of_segment(w1[knee_i], w1[ankle_i], 0.433)
            shank_acc  = acc[int((knee_i + ankle_i) / 2)]  # approx

            # Reaction force at ankle (ground reaction estimated)
            # For simplicity, estimate as body_weight × 1.2 during exercise
            grf = np.array([0.0, -self.mass * GRAVITY * 1.2])

            # Net force on shank segment
            net_force = shank_mass * shank_acc - grf - np.array([0.0, shank_mass * (-GRAVITY)])
            # Moment arm = perpendicular distance from knee to shank COM
            r = shank_com - w1[knee_i]
            torque_knee = float(r[0] * net_force[1] - r[1] * net_force[0])
            torques[f"{side}_knee"] = abs(torque_knee)

            # ── Hip torques ───────────────────────────────────────────────────
            thigh_mass = self.mass * SEGMENT_MASS_FRACTIONS["thigh"]
            thigh_com  = self._com_of_segment(w1[hip_i], w1[knee_i], 0.433)
            thigh_acc  = acc[int((hip_i + knee_i) / 2)]
            net_force_thigh = thigh_mass * thigh_acc
            r2 = thigh_com - w1[hip_i]
            torque_hip = float(r2[0] * net_force_thigh[1] - r2[1] * net_force_thigh[0])
            torques[f"{side}_hip"] = abs(torque_hip)

        # ── Shoulder torques ─────────────────────────────────────────────────
        for side in ("left", "right"):
            sh_i  = ji[f"{side}_shoulder"]
            el_i  = ji[f"{side}_elbow"]
            ua_mass = self.mass * SEGMENT_MASS_FRACTIONS["upper_arm"]
            ua_com  = self._com_of_segment(w1[sh_i], w1[el_i], 0.436)
            ua_acc  = acc[int((sh_i + el_i) / 2)]
            net_force_ua = ua_mass * ua_acc
            r3 = ua_com - w1[sh_i]
            torque_sh = float(r3[0] * net_force_ua[1] - r3[1] * net_force_ua[0])
            torques[f"{side}_shoulder"] = abs(torque_sh)

        # ── Normalise to stress index [0-100] ────────────────────────────────
        mass_scale = self.mass / 70.0
        for joint, t in torques.items():
            max_t = MAX_VOLUNTARY_TORQUES.get(joint, 100.0) * mass_scale
            stress[joint] = float(np.clip(t / max_t * 100.0, 0.0, 100.0))

        return torques, stress

    def _biomech_fatigue_score(self) -> float:
        """
        Biomechanical fatigue [0-100]:
        Rising mean stress index over time indicates mechanical fatigue accumulation.
        """
        if len(self._stress_history) < 5:
            return 0.0
        hist = np.array(self._stress_history)
        # Compare recent (last 20 %) vs early (first 20 %) stress
        n = len(hist)
        n5 = max(int(n * 0.2), 1)
        early  = np.mean(hist[:n5])
        recent = np.mean(hist[-n5:])
        # Normalise by early to get relative increase
        ratio = (recent - early) / (early + 1e-3)
        score = float(np.clip(ratio * 50.0, 0.0, 100.0))

        # Also factor in absolute cumulative load
        load_score = float(np.clip(self._cumulative_load / 5000.0 * 50.0, 0.0, 50.0))
        return float(np.clip(0.5 * score + 0.5 * load_score, 0.0, 100.0))