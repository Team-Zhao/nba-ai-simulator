# NBA AI Simulator

A NBA game prediction system built around dynamic player ratings, roster-based team features, and machine learning (inspired ny NBA 2K series).

This simulator converts historical NBA player performance into six-dimensional player ratings, aggregates those ratings into team-level matchup features. It then predicts both game outcomes and expected point differential.

## Version 1.0 MVP

V1 establishes the complete modeling pipeline:

```text
Historical Player Data
        ↓
Player Rating Engine
        ↓
Expected-Minutes Team Aggregation
        ↓
Game Matchup Features
        ↓
Machine Learning Models
        ↓
Win Probability + Predicted Margin
```

## Player Rating System

Each player is represented by six ratings:

- Finishing
- Shooting
- Playmaking
- Defense
- Rebounding
- Physical

The V1 rating engine uses historical box score and advanced statistics from the 2023-24 and 2024-25 NBA seasons via nba_api. These ratings are then used to construct team-level features for 2025-26 games.

## Matchup Features

V1 has simple matchup features. For each game, home and away team ratings are converted into six matchup differences:

- `finishingDiff`
- `shootingDiff`
- `playmakingDiff`
- `defenseDiff`
- `reboundingDiff`
- `physicalDiff`

## Model Training

The 2025-26 regular season provides 1,230 game-level samples.

Games are split chronologically:

```text
First 80% of season → Training
Last 20% of season  → Testing
```

This better reflects the real-world setting of using past games to predict future games.

V1 compares four models:

- Linear Regression
- Logistic Regression
- Random Forest
- XGBoost

## V1 Results

| Model | Winner Accuracy | Log Loss | Brier Score | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 78.5% | — | — | 12.15 | 15.49 |
| Logistic Regression | **79.7%** | **0.512** | **0.167** | — | — |
| Random Forest | 77.2% | 0.543 | 0.180 | — | — |
| XGBoost | 75.2% | 0.541 | 0.181 | — | — |

## License

This project is licensed under the MIT License.

NBA statistics used by this project are accessed through `nba_api` / NBA.com and are subject to the applicable NBA.com terms of use.

