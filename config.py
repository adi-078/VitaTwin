"""
⚙️ Global Configuration
Personalized Vision-Based Digital Twin Fatigue Intelligence System
"""
import os
from typing import Dict

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
PROFILES_DIR  = os.path.join(DATA_DIR, "profiles")
SESSIONS_DIR  = os.path.join(DATA_DIR, "sessions")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

for _d in [DATA_DIR, PROFILES_DIR, SESSIONS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
TARGET_FPS    = 30

# ─── MediaPipe Pose ────────────────────────────────────────────────────────────
POSE_MIN_DETECTION_CONFIDENCE = 0.7
POSE_MIN_TRACKING_CONFIDENCE  = 0.7
POSE_MODEL_COMPLEXITY         = 1       # 0=lite | 1=full | 2=heavy

# ─── MediaPipe Face Mesh ───────────────────────────────────────────────────────
FACE_MAX_FACES              = 1
FACE_REFINE_LANDMARKS       = True
FACE_MIN_DETECTION_CONF     = 0.7
FACE_MIN_TRACKING_CONF      = 0.7

# ─── Baseline Calibration ─────────────────────────────────────────────────────
CALIBRATION_DURATION_SEC    = 180       # 3 minutes of "fresh" movement capture
CALIBRATION_MIN_FRAMES      = 50        # minimum frames before baseline is usable
CALIBRATION_WARMUP_SEC      = 5         # discard first 5 s (camera / body settling)

# ─── Fatigue Estimation ────────────────────────────────────────────────────────
FATIGUE_WINDOW_SEC          = 10        # rolling window for feature aggregation
FATIGUE_UPDATE_HZ           = 5         # score emitted at 5 Hz
FATIGUE_SMOOTHING_ALPHA     = 0.20      # EMA factor (higher = more responsive)

SPEED_DROP_THRESHOLD        = 0.75      # speed < 75 % baseline → fatigued
ROM_DROP_THRESHOLD          = 0.80      # ROM  < 80 % baseline → fatigued
STABILITY_DROP_THRESHOLD    = 1.40      # jitter > 140 % baseline → fatigued

# ─── Cognitive Fatigue ─────────────────────────────────────────────────────────
# Eye Aspect Ratio  EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
# Reference: Soukupová & Čech (2016)
EAR_BLINK_THRESHOLD         = 0.20      # momentary dip → blink event
EAR_DROWSY_THRESHOLD        = 0.25      # sustained low EAR → drowsy
BLINK_CONSEC_FRAMES         = 3         # frames below threshold that count as a blink
BLINK_RATE_WINDOW_SEC       = 60        # compute blinks-per-minute over 60 s
NORMAL_BLINK_RATE           = 15        # blinks/min (population average)
HIGH_BLINK_RATE             = 25        # elevated → eye strain / fatigue
LOW_BLINK_RATE              = 8         # suppressed → deep focus (mild fatigue marker)

# ─── Digital Twin Biomechanics ─────────────────────────────────────────────────
DEFAULT_BODY_MASS_KG        = 70.0
SEGMENT_MASS_FRACTIONS: Dict[str, float] = {
    "thigh":     0.100,
    "shank":     0.0465,
    "foot":      0.0145,
    "upper_arm": 0.028,
    "forearm":   0.016,
    "hand":      0.006,
    "trunk":     0.497,
    "head":      0.081,
}
GRAVITY = 9.81
# Approximate segment lengths as fraction of body height
SEGMENT_LENGTH_FRACTIONS: Dict[str, float] = {
    "thigh":     0.245,
    "shank":     0.246,
    "foot":      0.152,
    "upper_arm": 0.186,
    "forearm":   0.146,
}

# ─── LSTM Predictor ────────────────────────────────────────────────────────────
LSTM_INPUT_DIM          = 12        # features fed to LSTM per timestep
LSTM_HIDDEN_DIM         = 64
LSTM_NUM_LAYERS         = 2
LSTM_DROPOUT            = 0.2
LSTM_SEQUENCE_LEN       = 150       # 30 s × 5 Hz
LSTM_PREDICTION_STEPS   = 60        # 2 min × 5 Hz  (prediction horizon)
LSTM_LEARNING_RATE      = 1e-3
LSTM_BATCH_SIZE         = 32
LSTM_EPOCHS             = 50
LSTM_MIN_SESSIONS       = 3         # sessions before personalized model trains
LSTM_MODEL_PATH         = os.path.join(MODELS_DIR, "lstm_fatigue.pth")

# ─── Fusion Weights ────────────────────────────────────────────────────────────
PHYSICAL_FATIGUE_WEIGHT  = 0.65
COGNITIVE_FATIGUE_WEIGHT = 0.35

# ─── Adaptive Coach ────────────────────────────────────────────────────────────
COACH_FEEDBACK_INTERVAL_SEC  = 30
FATIGUE_ALERT_THRESHOLD      = 65   # warn user
FATIGUE_CRITICAL_THRESHOLD   = 85   # recommend stop

# ─── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HISTORY_SEC  = 300        # 5 minutes of history on the timeline chart
OVERLAY_ALPHA          = 0.75

# ─── MediaPipe Landmark Indices ────────────────────────────────────────────────
JOINT_INDICES: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
    "left_heel":     29,  "right_heel":     30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# Face Mesh eye landmark indices (MediaPipe 468-point model)
# p1..p6 ordered per the EAR formula: p1=outer, p4=inner, p2/p3/p5/p6=vertical pairs
RIGHT_EYE_EAR_PTS = [33,  160, 158, 133, 153, 144]   # [p1,p2,p3,p4,p5,p6]
LEFT_EYE_EAR_PTS  = [362, 385, 387, 263, 373, 380]