"""Export the frozen held-out Team Logistic recipe for use in live inference.

Run from project root:
 PYTHONPATH=src uv run python -m nba_ai_simulator.models.export_team_logistic

Fits ONLY the original development rows (< 2026-01-30), and never reads
outcomes or features from held-out rows. Does not rewrite test results.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_ai_simulator.models.evaluate_final_player_model import build_team_differences
from nba_ai_simulator.models.player_model import get_player_feature_cols

DATA_PATH = Path('data/processed/player_level_game_features.csv')
OUTPUT_DIR = Path('models/player_level_final_v1/team_logistic')
CUTOFF = pd.Timestamp('2026-01-30')
EXPECTED_TRAIN_GAMES = 1944
RATING_NAMES = ['finishing', 'shooting', 'playmaking', 'defense', 'rebounding', 'physical']


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    model_path = OUTPUT_DIR / 'team_logistic.joblib'
    metadata_path = OUTPUT_DIR / 'metadata.json'
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(f'Team Logistic artifact exists: {OUTPUT_DIR}. No overwrite allowed.')

    # Only extract rows from the original training interval. This code never
    # constructs a test dataframe, and the test labels are never accessed.
    df = pd.read_csv(DATA_PATH, dtype={'gameId': str}, parse_dates=['gameDate'])
    if df['gameId'].duplicated().any():
        raise ValueError('Duplicated game IDs')
    train = df.loc[df['gameDate'] < CUTOFF].copy()
    if len(train) != EXPECTED_TRAIN_GAMES:
        raise ValueError(f'Expected {EXPECTED_TRAIN_GAMES} training games, got {len(train)}')
    if not np.isin(train['homeWin'], [0, 1]).all():
        raise ValueError('Invalid training labels')
    feature_cols = get_player_feature_cols(train.columns)
    x = build_team_differences(train)
    if x.shape != (EXPECTED_TRAIN_GAMES, len(RATING_NAMES)) or not np.isfinite(x).all():
        raise ValueError(f'Unexpected features: {x.shape}')
    y = train['homeWin'].to_numpy(dtype=np.int64)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000)),
    ])
    pipeline.fit(x, y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save a single pipeline object to avoid scaler/model mismatch.
    joblib.dump(pipeline, model_path)
    metadata = {
        'model_name': 'team_logistic_minutes_weighted_v1',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_cutoff_exclusive': str(CUTOFF.date()),
        'training_games': len(train),
        'rating_names': RATING_NAMES,
        'feature_representation': 'home_minus_away_minutes_weighted_six_ratings',
        'model': 'StandardScaler -> LogisticRegression(max_iter=2000)',
        'feature_order': feature_cols,
        'dataset_sha256': _sha256(DATA_PATH),
        'model_sha256': _sha256(model_path),
        'test_used_for_training': False,
    }
    with metadata_path.open('x') as fh:
        json.dump(metadata, fh, indent=2)
        fh.write('\n')
    print(f'Trained Team Logistic on {len(train)} development games only.')
    print('Saved:', model_path)
    print('Saved:', metadata_path)
    print('Sealed held-out test NOT re-evaluated.')


if __name__ == '__main__':
    main()
