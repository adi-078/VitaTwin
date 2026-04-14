"""
🧠🤖 Personalized Vision-Based Digital Twin Fatigue Intelligence System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main runtime loop.

Usage:
    python main.py                          # default user 'user1'
    python main.py --user alice             # named user
    python main.py --user alice --camera 1  # external webcam
    python main.py --user alice --calibrate # force re-calibration
    python main.py --user alice --rpe 7     # save session with RPE label

Controls:
    Q / ESC     → quit
    R           → reset calibration
    SPACE       → pause / resume
    S           → save session snapshot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import cv2
import argparse
import time
import uuid
import os
import sys
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    FATIGUE_UPDATE_HZ, LSTM_MIN_SESSIONS,
)

from modules.vision.pose_extractor      import PoseExtractor
from modules.vision.face_extractor      import FaceExtractor
from modules.personalization.baseline   import BaselineCalibrator
from modules.personalization.normalizer import FeatureNormalizer
from modules.personalization.profile    import UserProfile, SessionRecord
from modules.digital_twin.biomechanics  import BiomechanicsEngine
from modules.fatigue.physical           import PhysicalFatigueEstimator
from modules.fatigue.cognitive          import CognitiveFatigueEstimator
from modules.fatigue.fusion             import FatigueAggregator
from modules.prediction.lstm_model      import FatiguePredictor
from modules.coaching.coach             import AdaptiveCoach
from modules.dashboard.visualizer       import Dashboard


# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Personalized Digital Twin Fatigue Intelligence System"
    )
    p.add_argument("--user",       default="user1",     help="User ID")
    p.add_argument("--camera",     type=int, default=CAMERA_INDEX)
    p.add_argument("--calibrate",  action="store_true", help="Force new baseline calibration")
    p.add_argument("--rpe",        type=float, default=None,
                   help="RPE rating (1-10) to attach to this session on exit")
    p.add_argument("--mass",       type=float, default=70.0, help="Body mass in kg")
    p.add_argument("--height",     type=float, default=1.75, help="Body height in metres")
    p.add_argument("--no-face",    action="store_true", help="Disable cognitive fatigue tracking")
    p.add_argument("--width",      type=int, default=FRAME_WIDTH)
    p.add_argument("--height-px",  type=int, default=FRAME_HEIGHT)
    return p


def main():
    args = build_argparser().parse_args()

    print(f"\n🧠 Fatigue Intelligence System  —  User: {args.user}")
    print("━" * 60)

    # ── User profile ──────────────────────────────────────────────────────────
    profile = UserProfile(args.user)
    profile.set_anthropometrics(args.mass, args.height)
    print(f"  Sessions logged: {profile.session_count}")
    print(f"  Personal model ready: {profile.has_personal_model}")

    # ── Component initialisation ──────────────────────────────────────────────
    pose_extractor = PoseExtractor()
    face_extractor = FaceExtractor() if not args.no_face else None
    biomech_engine = BiomechanicsEngine(body_mass_kg=args.mass, height_m=args.height)

    calibrator = BaselineCalibrator()
    # Restore saved baseline unless forced recalibration
    if not args.calibrate and profile.baseline is not None and profile.baseline.is_ready:
        print("  ✓ Loaded saved baseline profile")
        normalizer = FeatureNormalizer(profile.baseline)
        calibration_done = True
    else:
        print("  ⏳ Starting baseline calibration (perform exercises naturally)…")
        normalizer = None
        calibration_done = False

    phys_estimator = PhysicalFatigueEstimator()
    cog_estimator  = CognitiveFatigueEstimator()
    aggregator     = FatigueAggregator()
    predictor      = FatiguePredictor(session_count=profile.session_count)
    coach          = AdaptiveCoach()
    dashboard      = Dashboard(frame_w=args.width, frame_h=args.height_px)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height_px)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    print(f"\n  Camera {args.camera} opened  ({args.width}×{args.height_px})")
    print("  Press Q/ESC to quit  |  R to re-calibrate  |  SPACE to pause")
    print("━" * 60 + "\n")

    # ── Timing ────────────────────────────────────────────────────────────────
    fatigue_interval  = 1.0 / FATIGUE_UPDATE_HZ
    last_fatigue_tick = 0.0
    paused            = False
    session_id        = str(uuid.uuid4())[:8]
    frame_dims        = (args.width, args.height_px)

    # State
    total_score    = None
    physical_score = None
    cog_score      = None
    prediction     = None
    coach_msgs     = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame read failed — retrying…")
                time.sleep(0.05)
                continue

            if paused:
                cv2.putText(frame, "PAUSED — press SPACE to resume",
                            (50, args.height_px // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (60, 200, 255), 2)
                cv2.imshow("Fatigue Intelligence System", frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord(' '):
                    paused = False
                continue

            # ── Pose extraction ───────────────────────────────────────────────
            pose_frame = pose_extractor.process_frame(frame)
            face_frame = face_extractor.process_frame(frame) if face_extractor else None

            # ── Skeleton overlay ──────────────────────────────────────────────
            frame = pose_extractor.draw_skeleton(frame, pose_frame)

            # ── Baseline calibration ──────────────────────────────────────────
            if not calibration_done:
                calibrator.update(pose_frame)
                if calibrator.is_complete:
                    bl = calibrator.get_profile()
                    profile.save_baseline(bl)
                    normalizer        = FeatureNormalizer(bl)
                    calibration_done  = True
                    print(f"  ✓ Baseline calibration complete ({bl.n_frames} frames)")
                    print(bl.summary())

            # ── Fatigue tick ──────────────────────────────────────────────────
            now = time.time()
            if calibration_done and normalizer and (now - last_fatigue_tick) >= fatigue_interval:
                last_fatigue_tick = now

                # Update normalizer
                normalizer.update(pose_frame)
                features = normalizer.get_features()

                # Digital twin biomechanics
                biomech = biomech_engine.process(pose_frame, frame_dims)

                # Fatigue estimation
                physical_score = phys_estimator.update(features, biomech)
                cog_score      = cog_estimator.update(face_frame)
                total_score    = aggregator.update(physical_score, cog_score)

                # Push to predictor
                predictor.push(features.feature_vector, total_score.total)
                if predictor.ready:
                    prediction = predictor.predict()

                # Coach feedback
                new_msgs = coach.update(total_score, physical_score, cog_score,
                                        features, prediction)
                if new_msgs:
                    coach_msgs = new_msgs
                    dashboard.push_messages(coach_msgs)
                    for m in new_msgs:
                        print(f"[Coach] {m}")

                dashboard.push_score(total_score.total)

            # ── Render dashboard ──────────────────────────────────────────────
            frame = dashboard.render(
                camera_frame=frame,
                total=total_score,
                physical=physical_score,
                cognitive=cog_score,
                prediction=prediction,
                calibrator=calibrator if not calibration_done else None,
            )

            cv2.imshow("Fatigue Intelligence System", frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                print("[System] Re-calibrating baseline…")
                calibrator.reset()
                calibration_done = False
                normalizer = None
                phys_estimator.reset()
                cog_estimator.reset()
                aggregator.reset()
                coach.reset_session()
                biomech_engine.reset_session()
            elif key == ord(' '):
                paused = True
            elif key == ord('s'):
                snap_path = f"snapshot_{session_id}_{int(now)}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[System] Snapshot saved → {snap_path}")

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
    finally:
        # ── Save session ──────────────────────────────────────────────────────
        if total_score is not None and aggregator.get_series().any():
            record = SessionRecord(
                session_id    = session_id,
                timestamp     = time.time(),
                duration_sec  = time.time() - coach._session_start,
                peak_fatigue  = aggregator.peak,
                mean_fatigue  = aggregator.mean,
                fatigue_curve = aggregator.downsample_to_60(),
                rpe_rating    = args.rpe,
            )
            profile.add_session(record)
            print(f"\n[System] Session {session_id} saved.")
            print(f"  Peak fatigue : {record.peak_fatigue:.1f}")
            print(f"  Mean fatigue : {record.mean_fatigue:.1f}")
            if args.rpe:
                print(f"  RPE label    : {args.rpe}")

        # ── Retrain LSTM if enough sessions ───────────────────────────────────
        if profile.session_count >= LSTM_MIN_SESSIONS:
            print(f"\n[System] {profile.session_count} sessions available — triggering LSTM training…")
            try:
                from modules.prediction.trainer import train as lstm_train
                history = profile.get_fatigue_history()
                if len(history) > 0:
                    series = history.flatten().astype(np.float32)
                    import numpy as np
                    delta = np.diff(series, prepend=series[0])
                    from config import LSTM_INPUT_DIM
                    fv = np.zeros((len(series), LSTM_INPUT_DIM), dtype=np.float32)
                    fv[:, 0] = series / 100.0
                    fv[:, 1] = delta / 10.0
                    from config import MODELS_DIR
                    user_model = os.path.join(MODELS_DIR, f"lstm_{args.user}.pth")
                    metrics = lstm_train(series, fv, user_model, verbose=True)
                    if metrics:
                        print(f"  LSTM MAE: {metrics.get('lstm_mae_points','N/A')} pts  "
                              f"vs naive: {metrics.get('naive_mae_points','N/A')} pts")
            except Exception as e:
                print(f"[WARNING] LSTM training failed: {e}")

        # ── Cleanup ───────────────────────────────────────────────────────────
        cap.release()
        pose_extractor.release()
        if face_extractor:
            face_extractor.release()
        cv2.destroyAllWindows()
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()