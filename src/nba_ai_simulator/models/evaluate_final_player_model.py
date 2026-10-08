"""One-time locked held-out evaluation: saved PyTorch model vs Team Logistic.

Run from the repository root:
    uv run python -m nba_ai_simulator.models.evaluate_final_player_model

This command uses outcomes only to evaluate the sealed test window. It never
fits or changes the saved PyTorch checkpoint. Existing output reports are not
overwritten, to encourage a single final evaluation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from nba_ai_simulator.models.player_model import (
    N_PLAYER_FEATURES,
    TOP_N_PLAYERS,
    get_player_feature_cols,
)
from nba_ai_simulator.models.predict_player_level import predict

DATA_PATH = Path("data/processed/player_level_game_features.csv")
MODEL_DIR = Path("models/player_level_final_v1")
REPORT_DIR = Path("models/player_level_final_v1/evaluation")
CUTOFF = "2026-01-30"
EXPECTED_TRAIN_GAMES = 1944
EXPECTED_TEST_GAMES = 516
TEAM_FEATURES = ("finishing", "shooting", "playmaking", "defense", "rebounding", "physical")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_team_differences(df: pd.DataFrame) -> np.ndarray:
    """Same six features as classical team baseline: minutes-weighted
    home minus away ratings, with unweighted mean for a zero-minute team.
    """
    arrays = []
    for side in ("home", "away"):
        ratings = np.stack(
            [
                np.stack(
                    [df[f"{side}_p{slot}_{feature}"].to_numpy(dtype=np.float32)
                     for feature in TEAM_FEATURES],
                    axis=1,
                )
                for slot in range(1, TOP_N_PLAYERS + 1)
            ], axis=1,
        )  # [games, 10, 6]
        minutes = np.stack(
            [df[f"{side}_p{slot}_expectedMinutesRosterAdjusted"].to_numpy(dtype=np.float32)
             for slot in range(1, TOP_N_PLAYERS + 1)], axis=1,
        )
        sums = minutes.sum(axis=1, keepdims=True)
        weighted = (ratings * (minutes / np.maximum(sums, 1e-6))[..., None]).sum(axis=1)
        fallback = ratings.mean(axis=1)
        arrays.append(np.where(sums > 0, weighted, fallback))
    return (arrays[0] - arrays[1]).astype(np.float32)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def main() -> None:
    report_path = REPORT_DIR / "evaluation_report.json"
    predictions_path = REPORT_DIR / "test_predictions.csv"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError(
            f"Evaluation output already exists in {REPORT_DIR}. "
            "Leave the original held-out report intact; do not overwrite it."
        )

    metadata_path = MODEL_DIR / "metadata.json"
    checkpoint_path = MODEL_DIR / "model.pt"
    scaler_path = MODEL_DIR / "player_scaler.joblib"
    for path in (DATA_PATH, metadata_path, checkpoint_path, scaler_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("train_cutoff_exclusive") != CUTOFF:
        raise ValueError("Checkpoint cutoff differs from frozen test cutoff")
    if metadata.get("training_games") != EXPECTED_TRAIN_GAMES:
        raise ValueError("Checkpoint training-game count differs from frozen protocol")
    if metadata.get("architecture") != "shared_player_encoder_fixed_minutes_pooling":
        raise ValueError("Unexpected checkpoint architecture")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("Checkpoint metadata does not indicate an untouched test")

    df = pd.read_csv(DATA_PATH, dtype={"gameId": str}, parse_dates=["gameDate"])
    feature_cols = get_player_feature_cols(df.columns)
    if metadata["player_feature_cols"] != feature_cols:
        raise ValueError("Player feature ordering does not match checkpoint")
    if df["gameId"].duplicated().any():
        raise ValueError("Duplicate game IDs detected")
    cutoff = pd.Timestamp(CUTOFF)
    train = df.loc[df["gameDate"] < cutoff].copy()
    test = df.loc[df["gameDate"] >= cutoff].copy()
    if (len(train), len(test)) != (EXPECTED_TRAIN_GAMES, EXPECTED_TEST_GAMES):
        raise ValueError(f"Unexpected chronological split: {len(train)} train, {len(test)} test")
    if train["gameDate"].max() >= test["gameDate"].min():
        raise ValueError("Chronological split has overlapping dates")
    for partition, name in ((train, "train"), (test, "test")):
        y = partition["homeWin"].to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"Invalid {name} homeWin labels")
    y_train = train["homeWin"].to_numpy(dtype=np.int64)
    y_test = test["homeWin"].to_numpy(dtype=np.int64)

    # The final PyTorch model is frozen. Only inference is performed.
    torch_predictions = predict(test, MODEL_DIR)
    torch_probs = torch_predictions["home_win_probability"].to_numpy(dtype=np.float64)

    # Fit a fresh, comparably defined team Logistic on development data only.
    X_train = build_team_differences(train)
    X_test = build_team_differences(test)
    scaler = StandardScaler().fit(X_train)
    baseline = LogisticRegression(max_iter=2000)
    baseline.fit(scaler.transform(X_train), y_train)
    baseline_probs = baseline.predict_proba(scaler.transform(X_test))[:, 1]

    if not (np.isfinite(torch_probs).all() and np.isfinite(baseline_probs).all()):
        raise ValueError("Predictions contain NaN or infinite values")
    if not ((0 <= torch_probs).all() and (torch_probs <= 1).all()):
        raise ValueError("PyTorch prediction probabilities out of range")

    scores = {
        "pytorch_shared_player_encoder": metrics(y_test, torch_probs),
        "team_logistic": metrics(y_test, baseline_probs),
    }
    output = pd.DataFrame({
        "gameId": test["gameId"].to_numpy(),
        "gameDate": test["gameDate"].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual_home_win": y_test,
        "pytorch_home_win_probability": torch_probs,
        "team_logistic_home_win_probability": baseline_probs,
    })
    for name in ("homeTeam", "awayTeam"):
        if name in test.columns:
            output[name] = test[name].to_numpy()
    report = {
        "evaluation_type": "one_time_chronological_held_out_test",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_date_inclusive_test": CUTOFF,
        "train_games": len(train),
        "test_games": len(test),
        "test_min_date": test["gameDate"].min().strftime("%Y-%m-%d"),
        "test_max_date": test["gameDate"].max().strftime("%Y-%m-%d"),
        "test_home_win_rate": float(y_test.mean()),
        "checkpoint_version": metadata["model_version"],
        "checkpoint_sha256": sha256(checkpoint_path),
        "player_scaler_sha256": sha256(scaler_path),
        "dataset_sha256": sha256(DATA_PATH),
        "baseline": "LogisticRegression(max_iter=2000), six minutes-weighted team rating differences, StandardScaler trained on development only",
        "scores": scores,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, never overwrite prior official results.
    with predictions_path.open("x") as file:
        output.to_csv(file, index=False)
    with report_path.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"Held-out evaluation: {len(train)} train / {len(test)} test")
    print(pd.DataFrame(scores).T.to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"Report: {report_path}")
    print(f"Per-game predictions: {predictions_path}")
    print("This is the sealed test report: do not tune and rerun on this test window.")


if __name__ == "__main__":
    main()
