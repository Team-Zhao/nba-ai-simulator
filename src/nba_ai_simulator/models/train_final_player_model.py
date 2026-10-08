"""Train a frozen final model on pre-cutoff games only; do not evaluate test data.

Run from the repository root:
    uv run python -m nba_ai_simulator.models.train_final_player_model
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nba_ai_simulator.models.player_model import (
    N_PLAYER_FEATURES, TOP_N_PLAYERS, SharedPlayerEncoderModel,
    fit_player_scaler, get_player_feature_cols, pregame_arrays, transform_players,
)

DATA_PATH = Path("data/processed/player_level_game_features.csv")
OUTPUT_DIR = Path("models/player_level_final_v1")
TRAIN_CUTOFF = "2026-01-30"  # EXCLUSIVE; locked before held-out test
SEED = 42
FINAL_EPOCHS = 10  # Preselected from median CV best epoch (15 runs)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_seed(SEED)
    df = pd.read_csv(DATA_PATH, dtype={"gameId": str}, parse_dates=["gameDate"])
    feature_cols = get_player_feature_cols(df.columns)
    train_df = df.loc[df["gameDate"] < pd.Timestamp(TRAIN_CUTOFF)].copy()
    # Deliberately do NOT create or inspect held-out test labels or metrics here.
    if len(train_df) != 1944:
        raise ValueError(f"Expected 1944 pre-cutoff games, got {len(train_df)}")
    if train_df["gameId"].nunique() != len(train_df):
        raise ValueError("Duplicate gameIds in training data")
    y = train_df["homeWin"].to_numpy(dtype=np.float32)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("homeWin must be binary 0/1")

    raw, minutes = pregame_arrays(train_df, feature_cols)
    scaler = fit_player_scaler(raw)  # fit on training data only
    X = torch.from_numpy(transform_players(raw, scaler))
    M = torch.from_numpy(minutes)
    Y = torch.from_numpy(y)

    loader = DataLoader(
        TensorDataset(X, M, Y), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = SharedPlayerEncoderModel(
        player_feature_dim=N_PLAYER_FEATURES, embedding_dim=16,
        players_per_team=TOP_N_PLAYERS,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, FINAL_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_minutes, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x, batch_minutes)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        print(f"Epoch {epoch:02d}/{FINAL_EPOCHS} | train BCE: {total_loss / len(train_df):.5f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), OUTPUT_DIR / "model.pt")
    joblib.dump(scaler, OUTPUT_DIR / "player_scaler.joblib")
    metadata = {
        "model_version": "player_level_final_v1",
        "architecture": "shared_player_encoder_fixed_minutes_pooling",
        "player_feature_dim": N_PLAYER_FEATURES,
        "embedding_dim": 16,
        "players_per_team": TOP_N_PLAYERS,
        "player_feature_cols": feature_cols,
        "train_cutoff_exclusive": TRAIN_CUTOFF,
        "training_games": int(len(train_df)),
        "training_min_date": train_df["gameDate"].min().strftime("%Y-%m-%d"),
        "training_max_date": train_df["gameDate"].max().strftime("%Y-%m-%d"),
        "training_rule": "10 fixed epochs; median best epoch from development-only temporal CV",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "probability_target": "homeWin",
        "test_evaluated": False,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved final checkpoint and preprocessing to {OUTPUT_DIR}")
    print("Held-out test set: NOT evaluated")


if __name__ == "__main__":
    main()
