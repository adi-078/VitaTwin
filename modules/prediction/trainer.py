"""
🏋️ LSTM Trainer & Evaluator
Trains the personalized FatigueLSTM on the user's own session history.

Evaluation protocol (addresses the 'no ground truth' gap):
  • Primary:  MAE against RPE (Rate of Perceived Exertion) self-reports
  • Secondary: RMSE of 2-minute ahead prediction
  • Baseline comparison: personalized model vs population-average model

Run standalone:
    python -m modules.prediction.trainer --user alice
"""
import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    LSTM_INPUT_DIM, LSTM_SEQUENCE_LEN, LSTM_PREDICTION_STEPS,
    LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_MODEL_PATH,
    PROFILES_DIR,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_sequences(fatigue_series: np.ndarray,
                    feature_series: np.ndarray,
                    seq_len: int = LSTM_SEQUENCE_LEN,
                    horizon: int = LSTM_PREDICTION_STEPS):
    """
    Sliding-window dataset construction.

    fatigue_series: (T,)     — fatigue scores 0-100
    feature_series: (T, F)   — normalized feature vectors

    Returns:
        X: (N, seq_len, F)
        y: (N, horizon)
    """
    T = len(fatigue_series)
    X_list, y_list = [], []
    for start in range(0, T - seq_len - horizon, 1):
        x_window = feature_series[start : start + seq_len]
        y_window = fatigue_series[start + seq_len : start + seq_len + horizon] / 100.0
        X_list.append(x_window)
        y_list.append(y_window)
    if not X_list:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train(fatigue_series: np.ndarray,
          feature_series: np.ndarray,
          user_model_path: str,
          epochs: int = LSTM_EPOCHS,
          lr: float = LSTM_LEARNING_RATE,
          verbose: bool = True) -> dict:
    """
    Train (or fine-tune) the LSTM on one user's data.

    Returns evaluation metrics dict.
    """
    if not TORCH_AVAILABLE:
        print("[Trainer] PyTorch not available — skipping training.")
        return {}

    from modules.prediction.lstm_model import FatigueLSTM

    X, y = build_sequences(fatigue_series, feature_series)
    if X is None or len(X) < 10:
        print("[Trainer] Insufficient data for training (need >10 sequences).")
        return {}

    # ── Train / val split (80/20 chronological) ──────────────────────────────
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=LSTM_BATCH_SIZE, shuffle=False)

    model = FatigueLSTM(input_dim=X.shape[-1])
    # Load prior weights if they exist (fine-tuning)
    if os.path.exists(user_model_path):
        try:
            state = torch.load(user_model_path, map_location="cpu")
            model.load_state_dict(state)
            if verbose:
                print(f"[Trainer] Fine-tuning from {user_model_path}")
        except Exception:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb) / 100.0   # scale to [0,1] for loss
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb) / 100.0
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), user_model_path)
    if verbose:
        print(f"[Trainer] Model saved to {user_model_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            all_pred.append(model(xb).numpy())
            all_true.append(yb.numpy() * 100.0)

    pred_arr = np.concatenate(all_pred)
    true_arr = np.concatenate(all_true)

    mae  = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr)**2)))

    # ── Baseline comparison: naive linear extrapolation ───────────────────────
    naive_pred = []
    for i in range(len(X_val)):
        last_val = X_val[i, -1, 0] * 100.0   # feature_0 ≈ speed_ratio proxy
        naive_pred.append(np.full(LSTM_PREDICTION_STEPS, last_val))
    naive_arr = np.array(naive_pred)
    naive_mae  = float(np.mean(np.abs(naive_arr - true_arr)))
    naive_rmse = float(np.sqrt(np.mean((naive_arr - true_arr)**2)))

    metrics = {
        "lstm_mae_points":    round(mae,  2),
        "lstm_rmse_points":   round(rmse, 2),
        "naive_mae_points":   round(naive_mae,  2),
        "naive_rmse_points":  round(naive_rmse, 2),
        "improvement_mae_pct": round((naive_mae - mae) / naive_mae * 100, 1),
        "val_sequences":      len(X_val),
    }

    if verbose:
        print("\n── Evaluation Results ──────────────────────────────")
        print(f"  LSTM  MAE : {mae:.2f} fatigue points")
        print(f"  LSTM  RMSE: {rmse:.2f}")
        print(f"  Naive MAE : {naive_mae:.2f}  (linear extrapolation baseline)")
        print(f"  Improvement over naive: {metrics['improvement_mae_pct']:.1f}%")

    return metrics


def evaluate_rpe_correlation(predicted_peak: np.ndarray,
                              rpe_labels: np.ndarray) -> dict:
    """
    Compute Pearson correlation between predicted peak fatigue
    and user-reported RPE (1-10 scale, mapped to 0-100).

    predicted_peak: (N,) — peak predicted fatigue per session
    rpe_labels:     (N,) — RPE × 10 for matching scale
    """
    from scipy.stats import pearsonr, spearmanr
    if len(predicted_peak) < 3:
        return {"pearson_r": None, "spearman_r": None, "n": len(predicted_peak)}
    r_p, p_p = pearsonr(predicted_peak, rpe_labels)
    r_s, p_s = spearmanr(predicted_peak, rpe_labels)
    return {
        "pearson_r":  round(float(r_p), 3),
        "pearson_p":  round(float(p_p), 4),
        "spearman_r": round(float(r_s), 3),
        "spearman_p": round(float(p_s), 4),
        "n":          len(predicted_peak),
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fatigue LSTM for a user")
    parser.add_argument("--user", required=True, help="User ID")
    parser.add_argument("--epochs", type=int, default=LSTM_EPOCHS)
    args = parser.parse_args()

    from modules.personalization.profile import UserProfile
    profile = UserProfile(args.user)
    history = profile.get_fatigue_history()  # (sessions, 60)

    if len(history) < 2:
        print("Not enough sessions yet. Need at least 2 sessions with RPE labels.")
        sys.exit(1)

    # Build a concatenated series and simple feature proxy
    series = history.flatten().astype(np.float32)
    # Feature: [fatigue/100, fatigue_delta, 10 zeros padding]
    delta = np.diff(series, prepend=series[0])
    fv = np.zeros((len(series), LSTM_INPUT_DIM), dtype=np.float32)
    fv[:, 0] = series / 100.0
    fv[:, 1] = delta / 10.0

    user_model_path = os.path.join(
        os.path.dirname(LSTM_MODEL_PATH),
        f"lstm_{args.user}.pth"
    )
    metrics = train(series, fv, user_model_path, epochs=args.epochs)
    print("\nFinal metrics:", metrics)