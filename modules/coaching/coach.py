"""
🤖 Adaptive AI Coach
Generates personalised, context-aware coaching feedback based on
the user's current fatigue state relative to THEIR baseline.

Feedback is never generic ("you are tired") — it is always specific
("your squat depth has dropped 18 % compared to your warm-up range").

Feedback types:
  • Technique cues     — joint-specific form corrections
  • Load warnings      — fatigue thresholds
  • Predictive alerts  — "fatigue likely to spike in ~2 min"
  • Recovery prompts   — rest suggestions
  • Encouragement      — positive reinforcement when improving
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    COACH_FEEDBACK_INTERVAL_SEC,
    FATIGUE_ALERT_THRESHOLD,
    FATIGUE_CRITICAL_THRESHOLD,
    FATIGUE_UPDATE_HZ,
)
from modules.fatigue.fusion import TotalFatigueScore
from modules.fatigue.physical import PhysicalFatigueScore
from modules.fatigue.cognitive import CognitiveFatigueScore
from modules.personalization.normalizer import NormalizedFeatures
from modules.prediction.lstm_model import PredictionResult


@dataclass
class CoachMessage:
    """A single piece of feedback from the coach."""
    text:       str
    category:   str   # "technique" | "warning" | "prediction" | "recovery" | "positive"
    urgency:    int   # 0=info, 1=mild, 2=important, 3=critical
    timestamp:  float = field(default_factory=time.time)

    def __str__(self):
        icons = {0: "💬", 1: "⚡", 2: "⚠️", 3: "🛑"}
        return f"{icons.get(self.urgency,'💬')} {self.text}"


class AdaptiveCoach:
    """
    Rule-based coaching engine with personalised thresholds.

    Respects a minimum feedback interval so it doesn't spam the user.
    Messages are queued and returned in priority order.

    Usage:
        coach = AdaptiveCoach()
        messages = coach.update(total_score, physical_score, cog_score,
                                norm_features, prediction)
        for msg in messages:
            print(msg)
    """

    def __init__(self):
        self._last_feedback_time: float = 0.0
        self._session_start:       float = time.time()
        self._prev_total:          float = 0.0
        self._peak_total:          float = 0.0
        self._improvement_streak:  int   = 0
        self._spoken_warnings:     set   = set()  # avoid repeating same warning

    # ──────────────────────────────────────────────────────────────────────────

    def update(self,
               total:    TotalFatigueScore,
               physical: PhysicalFatigueScore,
               cognitive: Optional[CognitiveFatigueScore],
               features:  NormalizedFeatures,
               prediction: Optional[PredictionResult] = None) -> List[CoachMessage]:
        """
        Evaluate current state and return a list of CoachMessages.
        Returns [] if the feedback interval has not elapsed.
        """
        now = time.time()
        messages: List[CoachMessage] = []

        # ── Critical override (always fires, regardless of interval) ──────────
        if total.critical and "critical" not in self._spoken_warnings:
            messages.append(CoachMessage(
                text="Your fatigue is critical. Please stop and rest now.",
                category="recovery", urgency=3,
            ))
            self._spoken_warnings.add("critical")

        # ── Respect feedback interval for non-critical messages ───────────────
        if (now - self._last_feedback_time) < COACH_FEEDBACK_INTERVAL_SEC:
            self._prev_total = total.total
            return messages

        self._last_feedback_time = now
        self._peak_total = max(self._peak_total, total.total)

        # ── Technique cues (specific joint-level feedback) ────────────────────
        messages += self._technique_cues(features, physical)

        # ── Load warnings ────────────────────────────────────────────────────
        messages += self._load_warnings(total)

        # ── Predictive alerts ────────────────────────────────────────────────
        if prediction is not None:
            messages += self._predictive_alerts(prediction, total)

        # ── Cognitive alerts ─────────────────────────────────────────────────
        if cognitive is not None:
            messages += self._cognitive_alerts(cognitive)

        # ── Positive reinforcement ────────────────────────────────────────────
        messages += self._positive_feedback(total)

        # ── Recovery suggestions ──────────────────────────────────────────────
        messages += self._recovery_suggestions(total, physical)

        self._prev_total = total.total
        return sorted(messages, key=lambda m: -m.urgency)[:3]  # max 3 per interval

    def reset_session(self):
        self._last_feedback_time  = 0.0
        self._session_start       = time.time()
        self._prev_total          = 0.0
        self._peak_total          = 0.0
        self._improvement_streak  = 0
        self._spoken_warnings.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Private sub-modules
    # ──────────────────────────────────────────────────────────────────────────

    def _technique_cues(self, f: NormalizedFeatures,
                        p: PhysicalFatigueScore) -> List[CoachMessage]:
        msgs = []

        # Speed
        if f.speed_ratio < 0.80:
            pct = round((1 - f.speed_ratio) * 100)
            msgs.append(CoachMessage(
                text=f"Your movement speed has dropped {pct}% below your warm-up baseline.",
                category="technique", urgency=1,
            ))

        # ROM per joint
        worst_joint = min(f.rom_ratio.items(), key=lambda kv: kv[1], default=None)
        if worst_joint and worst_joint[1] < 0.78:
            joint_name = worst_joint[0].replace("_", " ").title()
            pct = round((1 - worst_joint[1]) * 100)
            msgs.append(CoachMessage(
                text=f"Your {joint_name} range of motion has reduced by {pct}%. Focus on full extension.",
                category="technique", urgency=1,
            ))

        # Stability
        if f.stability_ratio > 1.5:
            msgs.append(CoachMessage(
                text="Your joint stability is decreasing. Slow down and focus on controlled movement.",
                category="technique", urgency=1,
            ))

        # Posture
        worst_posture = max(f.posture_deviation.items(), key=lambda kv: kv[1], default=None)
        if worst_posture and worst_posture[1] > 2.5:
            joint_name = worst_posture[0].replace("_", " ").title()
            msgs.append(CoachMessage(
                text=f"Your {joint_name} posture is drifting further than your personal norm. Reset your form.",
                category="technique", urgency=1,
            ))

        return msgs[:1]  # one technique cue per cycle

    def _load_warnings(self, total: TotalFatigueScore) -> List[CoachMessage]:
        msgs = []
        if total.alert and "alert" not in self._spoken_warnings:
            msgs.append(CoachMessage(
                text=f"Fatigue is building — score {total.total:.0f}/100. Consider reducing intensity.",
                category="warning", urgency=2,
            ))
            self._spoken_warnings.add("alert")
        return msgs

    def _predictive_alerts(self, pred: PredictionResult,
                           total: TotalFatigueScore) -> List[CoachMessage]:
        msgs = []
        if pred.trend == "rising" and pred.peak_predicted >= FATIGUE_ALERT_THRESHOLD:
            if pred.seconds_to_alert is not None:
                mins = round(pred.seconds_to_alert / 60, 1)
                msgs.append(CoachMessage(
                    text=f"Fatigue is predicted to reach alert level in ~{mins} min at your current pace.",
                    category="prediction", urgency=1,
                ))
            else:
                msgs.append(CoachMessage(
                    text=f"Fatigue trend is rising. Predicted peak: {pred.peak_predicted:.0f}/100.",
                    category="prediction", urgency=1,
                ))
        elif pred.trend == "stable":
            pass  # no alert for stable prediction
        return msgs

    def _cognitive_alerts(self, cog: CognitiveFatigueScore) -> List[CoachMessage]:
        msgs = []
        if cog.is_drowsy and "drowsy" not in self._spoken_warnings:
            msgs.append(CoachMessage(
                text=f"Signs of eye fatigue detected (EAR: {cog.mean_ear:.2f}). Consider a short mental break.",
                category="warning", urgency=2,
            ))
            self._spoken_warnings.add("drowsy")
        if cog.blink_rate_pm > 28:
            msgs.append(CoachMessage(
                text=f"Elevated blink rate ({cog.blink_rate_pm:.0f} bpm) suggests eye strain.",
                category="warning", urgency=1,
            ))
        return msgs[:1]

    def _positive_feedback(self, total: TotalFatigueScore) -> List[CoachMessage]:
        msgs = []
        delta = total.total - self._prev_total
        if delta < -5 and total.total < 50:
            self._improvement_streak += 1
            if self._improvement_streak >= 2:
                msgs.append(CoachMessage(
                    text="Great recovery! Your fatigue score is trending down. Keep this pace.",
                    category="positive", urgency=0,
                ))
        else:
            self._improvement_streak = 0
        return msgs

    def _recovery_suggestions(self, total: TotalFatigueScore,
                               physical: PhysicalFatigueScore) -> List[CoachMessage]:
        msgs = []
        session_min = (time.time() - self._session_start) / 60.0
        if total.total > 70 and session_min > 10:
            msgs.append(CoachMessage(
                text=f"You've been exercising for {session_min:.0f} min with high fatigue. A 2-minute rest is recommended.",
                category="recovery", urgency=2,
            ))
        elif physical.biomech_score > 60:
            msgs.append(CoachMessage(
                text="High joint load accumulation detected. Prioritise form over speed to reduce injury risk.",
                category="recovery", urgency=2,
            ))
        return msgs[:1]