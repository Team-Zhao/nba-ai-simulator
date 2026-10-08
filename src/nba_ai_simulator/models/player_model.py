"""Shared player encoder with fixed expected-minutes team pooling."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

PLAYER_FEATURES = (
    "finishing", "shooting", "playmaking", "defense", "rebounding",
    "physical", "expectedMinutesRosterAdjusted",
)
TOP_N_PLAYERS = 10
N_TEAMS = 2
N_PLAYER_FEATURES = len(PLAYER_FEATURES)


def get_player_feature_cols(columns) -> list[str]:
    available = set(columns)
    names = [
        f"{team}_p{slot}_{feature}"
        for team in ("home", "away")
        for slot in range(1, TOP_N_PLAYERS + 1)
        for feature in PLAYER_FEATURES
    ]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Missing {len(missing)} player features; examples: {missing[:5]}")
    return names


def pregame_arrays(df, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flat raw 140-D ratings/features and separate [N, 2, 10] raw minutes."""
    raw = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Player features contain NaN or infinite values")
    players = raw.reshape(-1, N_TEAMS, TOP_N_PLAYERS, N_PLAYER_FEATURES)
    minutes = players[..., -1].copy()
    if (minutes < 0).any():
        raise ValueError("Expected minutes cannot be negative")
    return raw, minutes


def fit_player_scaler(raw: np.ndarray) -> StandardScaler:
    """Fit one scaler per *feature*, shared across all 20 player slots."""
    return StandardScaler().fit(raw.reshape(-1, N_PLAYER_FEATURES))


def transform_players(raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(raw.reshape(-1, N_PLAYER_FEATURES))
    return scaled.astype(np.float32).reshape(-1, N_TEAMS * TOP_N_PLAYERS * N_PLAYER_FEATURES)


class PlayerEncoder(nn.Module):
    def __init__(self, player_feature_dim: int = N_PLAYER_FEATURES, embedding_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(player_feature_dim, 32), nn.ReLU(),
            nn.Linear(32, embedding_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SharedPlayerEncoderModel(nn.Module):
    def __init__(self, player_feature_dim: int = N_PLAYER_FEATURES,
                 embedding_dim: int = 16, players_per_team: int = TOP_N_PLAYERS):
        super().__init__()
        self.player_feature_dim = player_feature_dim
        self.embedding_dim = embedding_dim
        self.players_per_team = players_per_team
        self.player_encoder = PlayerEncoder(player_feature_dim, embedding_dim)
        self.game_network = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, raw_minutes: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        players = x.reshape(batch_size, 2, self.players_per_team, self.player_feature_dim)
        minutes = raw_minutes.reshape(batch_size, 2, self.players_per_team)
        embeddings = self.player_encoder(players)
        denominators = minutes.sum(dim=2, keepdim=True).clamp(min=1e-6)
        weights = minutes / denominators
        teams = (embeddings * weights.unsqueeze(-1)).sum(dim=2)
        game_embedding = torch.cat([teams[:, 0, :], teams[:, 1, :]], dim=1)
        return self.game_network(game_embedding).squeeze(1)
