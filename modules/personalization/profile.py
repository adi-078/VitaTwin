"""
👤 Personalization Engine — User Profile
Persists baseline profiles and session history to disk.
Handles cold-start (first session) gracefully.
"""
import json
import os
import time
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import PROFILES_DIR, SESSIONS_DIR, LSTM_MIN_SESSIONS
from modules.personalization.baseline import BaselineProfile


@dataclass
class SessionRecord:
    """Summary statistics saved after each completed session."""
    session_id:    str
    timestamp:     float
    duration_sec:  float
    peak_fatigue:  float
    mean_fatigue:  float
    fatigue_curve: List[float]   # downsampled to 60 points
    rpe_rating:    Optional[float] = None   # user self-report (1–10)
    notes:         str = ""


class UserProfile:
    """
    Manages everything we know about a specific user across sessions.

    Responsibilities:
      1. Load / save baseline to disk.
      2. Store session records for LSTM training.
      3. Answer cold-start queries (is there a trained personal model?).

    Usage:
        profile = UserProfile("alice")
        baseline = profile.baseline          # loads from disk or returns None
        profile.save_baseline(new_baseline)
        profile.add_session(record)
    """

    def __init__(self, user_id: str):
        self.user_id       = user_id
        self._profile_path = os.path.join(PROFILES_DIR, f"{user_id}.json")
        self._sessions_dir = os.path.join(SESSIONS_DIR, user_id)
        os.makedirs(self._sessions_dir, exist_ok=True)

        self._baseline: Optional[BaselineProfile] = None
        self._sessions: List[SessionRecord]        = []
        self._load()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def baseline(self) -> Optional[BaselineProfile]:
        return self._baseline

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def has_personal_model(self) -> bool:
        """True once enough sessions have been recorded to train a personal LSTM."""
        return self.session_count >= LSTM_MIN_SESSIONS

    @property
    def body_mass_kg(self) -> float:
        return self._meta.get("body_mass_kg", 70.0)

    @property
    def height_m(self) -> float:
        return self._meta.get("height_m", 1.75)

    def set_anthropometrics(self, mass_kg: float, height_m: float) -> None:
        self._meta["body_mass_kg"] = mass_kg
        self._meta["height_m"]     = height_m
        self._save_meta()

    def save_baseline(self, baseline: BaselineProfile) -> None:
        """Persist the latest baseline to disk (replaces previous)."""
        self._baseline = baseline
        self._save_meta()

    def add_session(self, record: SessionRecord) -> None:
        """Append a completed session record and persist it."""
        self._sessions.append(record)
        path = os.path.join(self._sessions_dir, f"{record.session_id}.json")
        with open(path, "w") as f:
            data = {
                "session_id":   record.session_id,
                "timestamp":    record.timestamp,
                "duration_sec": record.duration_sec,
                "peak_fatigue": record.peak_fatigue,
                "mean_fatigue": record.mean_fatigue,
                "fatigue_curve": record.fatigue_curve,
                "rpe_rating":   record.rpe_rating,
                "notes":        record.notes,
            }
            json.dump(data, f, indent=2)

    def get_recent_sessions(self, n: int = 10) -> List[SessionRecord]:
        return self._sessions[-n:]

    def get_fatigue_history(self) -> np.ndarray:
        """
        Returns a 2-D array (n_sessions × 60) of fatigue curves
        for LSTM training.  Only sessions with RPE labels included.
        """
        curves = [s.fatigue_curve for s in self._sessions
                  if s.rpe_rating is not None and len(s.fatigue_curve) == 60]
        if not curves:
            return np.empty((0, 60), dtype=np.float32)
        return np.array(curves, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        self._meta: Dict = {}
        if not os.path.exists(self._profile_path):
            return

        with open(self._profile_path) as f:
            data = json.load(f)

        self._meta = data.get("meta", {})

        bl_data = data.get("baseline")
        if bl_data:
            self._baseline = BaselineProfile(
                speed_mean      = bl_data["speed_mean"],
                speed_std       = bl_data["speed_std"],
                rom_mean        = bl_data["rom_mean"],
                rom_std         = bl_data["rom_std"],
                stability_mean  = bl_data["stability_mean"],
                stability_std   = bl_data["stability_std"],
                angle_mean      = bl_data["angle_mean"],
                angle_std       = bl_data["angle_std"],
                n_frames        = bl_data["n_frames"],
                capture_time    = bl_data["capture_time"],
                is_ready        = bl_data["is_ready"],
            )

        # Load session index
        for fname in sorted(os.listdir(self._sessions_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self._sessions_dir, fname)) as f:
                        s = json.load(f)
                    self._sessions.append(SessionRecord(**s))
                except Exception:
                    pass

    def _save_meta(self) -> None:
        bl = self._baseline
        data = {
            "user_id": self.user_id,
            "meta": self._meta,
            "baseline": None if bl is None else {
                "speed_mean":     bl.speed_mean,
                "speed_std":      bl.speed_std,
                "rom_mean":       bl.rom_mean,
                "rom_std":        bl.rom_std,
                "stability_mean": bl.stability_mean,
                "stability_std":  bl.stability_std,
                "angle_mean":     bl.angle_mean,
                "angle_std":      bl.angle_std,
                "n_frames":       bl.n_frames,
                "capture_time":   bl.capture_time,
                "is_ready":       bl.is_ready,
            },
        }
        with open(self._profile_path, "w") as f:
            json.dump(data, f, indent=2)