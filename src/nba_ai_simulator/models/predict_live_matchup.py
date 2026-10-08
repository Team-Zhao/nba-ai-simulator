"""Predict one upcoming matchup from prebuilt, full, pregame roster CSVs.

Does not fetch injury reports, alter stored models, or use held-out labels.
Run from project root:
 PYTHONPATH=src uv run python -m nba_ai_simulator.models.predict_live_matchup \
   --home data/processed/live_rosters/2026-27_NYK_2026-10-08.csv \
   --away data/processed/live_rosters/2026-27_WAS_2026-10-08.csv \
   --home-team NYK --away-team WAS --date 2026-10-08
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_ai_simulator.features.build_live_matchup_features import build_live_matchup_features
from nba_ai_simulator.models.evaluate_final_player_model import build_team_differences
from nba_ai_simulator.models.player_model import get_player_feature_cols
from nba_ai_simulator.models.predict_player_level import predict as predict_pytorch

DEFAULT_MODEL_DIR = Path('models/player_level_final_v1')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def predict_matchup(home: pd.DataFrame, away: pd.DataFrame, *, home_team: str,
                    away_team: str, game_date: str, game_id: str = 'demo',
                    model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    """Use immutable trained artifacts on raw roster features. Only trusted joblib files."""
    for side, roster, team in [('home', home, home_team), ('away', away, away_team)]:
        if 'teamTricode' in roster and not roster['teamTricode'].eq(team).all():
            raise ValueError(f'{side}: teamTricode mismatch for {team}')
    features = build_live_matchup_features(home, away, game_id=game_id,
                                          game_date=game_date, home_team=home_team,
                                          away_team=away_team)
    logistic_path = model_dir / 'team_logistic' / 'team_logistic.joblib'
    logistic_meta_path = model_dir / 'team_logistic' / 'metadata.json'
    metadata = json.loads(logistic_meta_path.read_text())
    if metadata.get('training_games') != 1944 or metadata.get('training_cutoff_exclusive') != '2026-01-30':
        raise ValueError('Unexpected Team Logistic training protocol')
    if metadata.get('model_sha256') != _sha256(logistic_path):
        raise ValueError('Team Logistic artifact hash mismatch')
    pipeline = joblib.load(logistic_path)  # only trusted locally generated model artifacts
    if list(pipeline.named_steps) != ['scaler', 'classifier']:
        raise ValueError('Unexpected Team Logistic pipeline')
    expected_cols = get_player_feature_cols(features.baseline.columns)
    if metadata.get('feature_order') != expected_cols:
        raise ValueError('Logistic feature schema differs from live adapter')

    results = {}
    for label, frame in [('baseline', features.baseline), ('injury_adjusted', features.injury_adjusted)]:
        x_team = build_team_differences(frame)
        p_log = float(pipeline.predict_proba(x_team)[0, 1])
        p_nn = float(predict_pytorch(frame, model_dir=model_dir)['home_win_probability'].iloc[0])
        for val in (p_log, p_nn):
            if not np.isfinite(val) or not 0 <= val <= 1:
                raise ValueError('Invalid model probability')
        results[label] = {
            'team_logistic_home_win_probability': p_log,
            'pytorch_home_win_probability': p_nn,
            'team_logistic_predicted_winner': home_team if p_log >= 0.5 else away_team,
            'pytorch_predicted_winner': home_team if p_nn >= 0.5 else away_team,
            'home_top10': features.baseline_rosters['home']['personId'].astype(int).tolist() if label == 'baseline' else features.adjusted_rosters['home']['personId'].astype(int).tolist(),
            'away_top10': features.baseline_rosters['away']['personId'].astype(int).tolist() if label == 'baseline' else features.adjusted_rosters['away']['personId'].astype(int).tolist(),
        }
    return {
        'game_id': game_id,
        'game_date': game_date,
        'home_team': home_team,
        'away_team': away_team,
        'data_quality': 'UNVERIFIED - review full roster, historical minutes, injury report and 2026 preseason shift',
        'rating_freshness_note': 'Historical rolling ratings are no newer than available local inputs; inspect ratingsAsOf/minutesAsOf.',
        'injury_adjustment_note': 'Heuristic scenario, not a separately validated model. UNKNOWN remains unadjusted.',
        'predictions': results,
        'feature_diagnostics': features.diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=Path, required=True, help='Full home-roster feature CSV')
    parser.add_argument('--away', type=Path, required=True, help='Full away-roster feature CSV')
    parser.add_argument('--home-team', required=True)
    parser.add_argument('--away-team', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--game-id', default='demo')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--output', type=Path, help='Optional output JSON path, refuses overwrite')
    args = parser.parse_args()
    output = predict_matchup(pd.read_csv(args.home), pd.read_csv(args.away),
                             home_team=args.home_team, away_team=args.away_team,
                             game_date=args.date, game_id=args.game_id,
                             model_dir=args.model_dir)
    result = json.dumps(output, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            f.write(result + '\n')
        print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
