"""
📊 Explainable Dashboard
Real-time OpenCV overlay with:
  • Live fatigue score gauge
  • Physical + cognitive breakdown bars
  • 5-minute fatigue timeline
  • Predicted future fatigue curve
  • Personalised insight text (contributing factors)
  • Coach message display
  • Calibration progress bar
"""
import cv2
import numpy as np
from typing import Optional, List, Deque
from collections import deque
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    DASHBOARD_HISTORY_SEC,
    FATIGUE_UPDATE_HZ,
    FATIGUE_ALERT_THRESHOLD,
    FATIGUE_CRITICAL_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT,
)
from modules.fatigue.fusion import TotalFatigueScore
from modules.fatigue.physical import PhysicalFatigueScore
from modules.fatigue.cognitive import CognitiveFatigueScore
from modules.prediction.lstm_model import PredictionResult
from modules.coaching.coach import CoachMessage
from modules.personalization.baseline import BaselineCalibrator


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_GREEN   = (40, 210, 100)
C_AMBER   = (30, 170, 240)
C_RED     = (50,  60, 230)
C_BLUE    = (220, 160,  40)
C_WHITE   = (240, 240, 240)
C_DARK    = ( 20,  20,  25)
C_PANEL   = ( 28,  28,  35)
C_BORDER  = ( 60,  60,  75)
C_TEXT    = (220, 220, 230)
C_MUTED   = (130, 130, 145)
C_PURPLE  = (200,  80, 180)
C_TEAL    = (200, 210,  60)


def _score_colour(score: float):
    if score < FATIGUE_ALERT_THRESHOLD:
        return C_GREEN
    if score < FATIGUE_CRITICAL_THRESHOLD:
        return C_AMBER
    return C_RED


class Dashboard:
    """
    Composites the full HUD over the live camera feed.

    Usage:
        dash = Dashboard(frame_w=1280, frame_h=720)
        output = dash.render(camera_frame, total_score, physical_score,
                             cog_score, prediction, messages, calibrator)
    """

    HISTORY_STEPS = int(DASHBOARD_HISTORY_SEC * FATIGUE_UPDATE_HZ)
    PANEL_W       = 340
    PANEL_ALPHA   = 0.82

    def __init__(self, frame_w: int = FRAME_WIDTH, frame_h: int = FRAME_HEIGHT):
        self.fw = frame_w
        self.fh = frame_h
        self._fatigue_history: Deque[float] = deque(maxlen=self.HISTORY_STEPS)
        self._coach_messages:  List[CoachMessage] = []
        self._coach_timer:     float = 0.0
        self._session_start:   float = time.time()

    # ──────────────────────────────────────────────────────────────────────────

    def push_score(self, score: float) -> None:
        self._fatigue_history.append(score)

    def push_messages(self, messages: List[CoachMessage]) -> None:
        if messages:
            self._coach_messages = messages
            self._coach_timer = time.time()

    def render(self,
               camera_frame: np.ndarray,
               total:        Optional[TotalFatigueScore],
               physical:     Optional[PhysicalFatigueScore],
               cognitive:    Optional[CognitiveFatigueScore],
               prediction:   Optional[PredictionResult],
               calibrator:   Optional[BaselineCalibrator] = None) -> np.ndarray:
        """Render the full dashboard overlay and return the composited frame."""
        out = camera_frame.copy()
        out = self._draw_panel(out, total, physical, cognitive, prediction)
        out = self._draw_timeline(out, prediction)
        out = self._draw_coach_overlay(out)
        if calibrator and not (calibrator.is_complete):
            out = self._draw_calibration_bar(out, calibrator)
        out = self._draw_session_info(out, total)
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Panel sections
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_panel(self, frame: np.ndarray,
                    total:    Optional[TotalFatigueScore],
                    physical: Optional[PhysicalFatigueScore],
                    cognitive: Optional[CognitiveFatigueScore],
                    prediction: Optional[PredictionResult]) -> np.ndarray:
        """Right-side dark panel with all score details."""
        pw   = self.PANEL_W
        ph   = self.fh
        px   = self.fw - pw

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, 0), (self.fw, ph), C_PANEL, -1)
        cv2.addWeighted(overlay, self.PANEL_ALPHA, frame, 1 - self.PANEL_ALPHA, 0, frame)
        cv2.line(frame, (px, 0), (px, ph), C_BORDER, 1)

        y = 20
        # ── Header ────────────────────────────────────────────────────────────
        cv2.putText(frame, "FATIGUE INTELLIGENCE", (px + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1, cv2.LINE_AA)
        y += 35
        cv2.line(frame, (px + 10, y), (self.fw - 10, y), C_BORDER, 1)
        y += 15

        # ── Main gauge ────────────────────────────────────────────────────────
        score = total.total if total else 0.0
        colour = _score_colour(score)
        cv2.putText(frame, f"{score:.0f}", (px + 25, y + 60),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, colour, 3, cv2.LINE_AA)
        cv2.putText(frame, "/ 100", (px + 125, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, "TOTAL FATIGUE", (px + 25, y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_MUTED, 1, cv2.LINE_AA)
        y += 100

        # ── Physical / cognitive split ────────────────────────────────────────
        if total:
            cv2.putText(frame, f"Physical  {total.physical:.0f}", (px + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEAL, 1, cv2.LINE_AA)
            self._bar(frame, px + 130, y - 12, 180, 14, total.physical / 100, C_TEAL)

            cv2.putText(frame, f"Cognitive {total.cognitive:.0f}", (px + 15, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_PURPLE, 1, cv2.LINE_AA)
            self._bar(frame, px + 130, y + 10, 180, 14, total.cognitive / 100, C_PURPLE)
        y += 55

        cv2.line(frame, (px + 10, y), (self.fw - 10, y), C_BORDER, 1)
        y += 15

        # ── Contributing factors ──────────────────────────────────────────────
        cv2.putText(frame, "WHY — YOUR BASELINE COMPARISON", (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1, cv2.LINE_AA)
        y += 15

        if physical:
            factors = [
                ("Speed drop",       physical.speed_score),
                ("ROM reduction",    physical.rom_score),
                ("Instability",      physical.stability_score),
                ("Posture drift",    physical.posture_score),
                ("Joint stress",     physical.biomech_score),
            ]
            for label, val in factors:
                c = _score_colour(val)
                cv2.putText(frame, f"{label}", (px + 15, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)
                self._bar(frame, px + 145, y, 160, 12, val / 100, c)
                cv2.putText(frame, f"{val:.0f}", (px + 310, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
                y += 20

        if cognitive:
            cv2.putText(frame, f"EAR: {cognitive.mean_ear:.2f}  Blinks/min: {cognitive.blink_rate_pm:.0f}",
                        (px + 15, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_PURPLE, 1, cv2.LINE_AA)
            y += 20

        y += 5
        cv2.line(frame, (px + 10, y), (self.fw - 10, y), C_BORDER, 1)
        y += 15

        # ── Prediction summary ────────────────────────────────────────────────
        if prediction:
            trend_icons = {"rising": "↑ RISING", "stable": "→ STABLE", "falling": "↓ FALLING"}
            trend_colours = {"rising": C_RED, "stable": C_AMBER, "falling": C_GREEN}
            trend_str = trend_icons.get(prediction.trend, "→")
            trend_col = trend_colours.get(prediction.trend, C_WHITE)
            cv2.putText(frame, "2-MIN PREDICTION", (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1, cv2.LINE_AA)
            y += 16
            cv2.putText(frame, trend_str, (px + 15, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, trend_col, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Peak: {prediction.peak_predicted:.0f}  Method: {prediction.method[:4]}",
                        (px + 130, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_MUTED, 1, cv2.LINE_AA)
            if prediction.seconds_to_alert is not None:
                cv2.putText(frame,
                            f"Alert in ~{prediction.seconds_to_alert/60:.1f} min",
                            (px + 15, y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_AMBER, 1, cv2.LINE_AA)

        return frame

    def _draw_timeline(self, frame: np.ndarray,
                       prediction: Optional[PredictionResult]) -> np.ndarray:
        """Bottom timeline: historical fatigue + predicted future."""
        TH = 120     # timeline height
        TY = self.fh - TH
        TW = self.fw - self.PANEL_W
        PAD = 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, TY), (TW, self.fh), C_PANEL, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, TY), (TW, TY), C_BORDER, 1)

        cv2.putText(frame, "FATIGUE TIMELINE  (last 5 min + 2 min forecast)",
                    (PAD, TY + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_MUTED, 1, cv2.LINE_AA)

        # Grid lines at 25, 50, 75, 100
        chart_y0 = TY + 28
        chart_h  = TH - 38
        chart_x0 = PAD + 35
        chart_w  = TW - PAD * 2 - 35

        for pct in [0.25, 0.50, 0.75, 1.00]:
            gy = int(chart_y0 + chart_h * (1 - pct))
            cv2.line(frame, (chart_x0, gy), (chart_x0 + chart_w, gy), C_BORDER, 1)
            cv2.putText(frame, f"{int(pct*100)}", (PAD, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_MUTED, 1, cv2.LINE_AA)

        # History line
        hist = list(self._fatigue_history)
        hist_w = int(chart_w * 0.7)
        if len(hist) >= 2:
            pts = []
            for i, v in enumerate(hist):
                x = chart_x0 + int(i / max(len(hist) - 1, 1) * hist_w)
                y = int(chart_y0 + chart_h * (1 - v / 100))
                pts.append((x, y))
            for i in range(1, len(pts)):
                colour = _score_colour(hist[i])
                cv2.line(frame, pts[i - 1], pts[i], colour, 2, cv2.LINE_AA)
            # Current dot
            cv2.circle(frame, pts[-1], 4, _score_colour(hist[-1]), -1)

        # Prediction line (dashed, lighter)
        if prediction is not None:
            pred = prediction.forecast
            pred_x0 = chart_x0 + hist_w
            pred_w  = chart_w - hist_w
            prev_pt = None
            for i in range(0, len(pred), max(1, len(pred)//40)):
                v = float(pred[i])
                x = pred_x0 + int(i / max(len(pred) - 1, 1) * pred_w)
                y = int(chart_y0 + chart_h * (1 - v / 100))
                pt = (x, y)
                if prev_pt and i % 4 < 2:   # dash pattern
                    cv2.line(frame, prev_pt, pt, C_AMBER, 1, cv2.LINE_AA)
                prev_pt = pt
            cv2.putText(frame, "← history     forecast →",
                        (chart_x0 + hist_w - 50, TY + TH - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_MUTED, 1, cv2.LINE_AA)

        # Divider between history and forecast
        cv2.line(frame,
                 (chart_x0 + hist_w, chart_y0),
                 (chart_x0 + hist_w, chart_y0 + chart_h),
                 C_BORDER, 1)

        return frame

    def _draw_coach_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Show coach messages in a semi-transparent bar."""
        if not self._coach_messages:
            return frame
        if (time.time() - self._coach_timer) > 15.0:
            self._coach_messages = []
            return frame

        msg_colours = {0: C_GREEN, 1: C_TEAL, 2: C_AMBER, 3: C_RED}
        y_base = self.fh - 125 - 30 * len(self._coach_messages)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_base - 10),
                      (self.fw - self.PANEL_W, y_base + 30 * len(self._coach_messages) + 5),
                      C_DARK, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        for i, msg in enumerate(self._coach_messages):
            c = msg_colours.get(msg.urgency, C_WHITE)
            cv2.putText(frame, str(msg), (15, y_base + i * 30 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 1, cv2.LINE_AA)
        return frame

    def _draw_calibration_bar(self, frame: np.ndarray,
                              cal: BaselineCalibrator) -> np.ndarray:
        """Top banner during baseline calibration phase."""
        progress = cal.progress
        bar_w    = int((self.fw - self.PANEL_W) * progress)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.fw - self.PANEL_W, 40), C_DARK, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (0, 0), (bar_w, 40), C_TEAL, -1)
        cv2.putText(frame,
                    f"  Calibrating your personal baseline... {progress*100:.0f}%  "
                    f"({cal.frames_collected} frames)",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_DARK if progress > 0.15 else C_WHITE,
                    1, cv2.LINE_AA)
        return frame

    def _draw_session_info(self, frame: np.ndarray,
                           total: Optional[TotalFatigueScore]) -> np.ndarray:
        elapsed = time.time() - self._session_start
        m, s    = divmod(int(elapsed), 60)
        cv2.putText(frame, f"{m:02d}:{s:02d}", (10, self.fh - 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_MUTED, 1, cv2.LINE_AA)
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _bar(frame: np.ndarray, x: int, y: int, w: int, h: int,
             fraction: float, colour: tuple) -> None:
        """Draw a horizontal progress bar."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), C_BORDER, -1)
        filled = int(w * max(0.0, min(1.0, fraction)))
        if filled > 0:
            cv2.rectangle(frame, (x, y), (x + filled, y + h), colour, -1)