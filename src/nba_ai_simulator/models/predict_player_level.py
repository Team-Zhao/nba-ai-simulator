"""Load a saved final model and predict from pregame player features.

Usage (from repository root):
    uv run python -m nba_ai_simulator.models.predict_player_level --input path/to/pregame.csv

Input file: one row per game, including all 140 player feature columns.
No observed game outcomes are read or required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from nba_ai_simulator.models.player_model import (
    SharedPlayerEncoderModel, pregame_arrays, transform_players,
)

DEFAULT_DIR = Path("models/player_level_final_v1")


def predict(df: pd.DataFrame, model_dir: Path = DEFAULT_DIR) -> pd.DataFrame:
    metadata = json.loads((model_dir / "metadata.json").read_text())
    feature_cols = metadata["player_feature_cols"]
    if len(feature_cols) != 140:
        raise ValueError("Unexpected checkpoint feature count")
    scaler = joblib.load(model_dir / "player_scaler.joblib")  # trusted local artifacts only
    model = SharedPlayerEncoderModel(
        player_feature_dim=metadata["player_feature_dim"],
        embedding_dim=metadata["embedding_dim"],
        players_per_team=metadata["players_per_team"],
    )
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    raw, raw_minutes = pregame_arrays(df, feature_cols)
    x = torch.from_numpy(transform_players(raw, scaler))
    minutes = torch.from_numpy(raw_minutes)
    with torch.inference_mode():
        home_probs = torch.sigmoid(model(x, minutes)).numpy()
    result = pd.DataFrame({
        "home_win_probability": home_probs,
        "away_win_probability": 1 - home_probs,
        "predicted_winner": np.where(home_probs >= 0.5, "home", "away"),
    })
    for key in ("gameId", "gameDate", "homeTeam", "awayTeam"):
        if key in df.columns:
            result.insert(0, key, df[key].to_numpy())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Pregame feature CSV")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    print(predict(pd.read_csv(args.input), args.model_dir).to_string(index=False))


if __name__ == "__main__":
    main()
