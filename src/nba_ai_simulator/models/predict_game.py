from pathlib import Path

import joblib
import pandas as pd


MODEL_DIR = Path("models")

LINEAR_MODEL_PATH = (
    MODEL_DIR
    / "linear_game_model_v1.joblib"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIR
    / "logistic_game_model_v1.joblib"
)

FEATURE_COLS = [
    "finishingDiff",
    "shootingDiff",
    "playmakingDiff",
    "defenseDiff",
    "reboundingDiff",
    "physicalDiff",
]

GAME_FEATURES_PATH = (
    Path("data/processed")
    / "game_features_2025_26.csv"
)

TEAM_RATINGS = [
    "Finishing",
    "Shooting",
    "Playmaking",
    "Defense",
    "Rebounding",
    "Physical",
]

linear_model = joblib.load(
    LINEAR_MODEL_PATH
)

logistic_model = joblib.load(
    LOGISTIC_MODEL_PATH
)

def predict_from_features(
    finishing_diff,
    shooting_diff,
    playmaking_diff,
    defense_diff,
    rebounding_diff,
    physical_diff,
):
    features = pd.DataFrame(
        [{
            "finishingDiff": finishing_diff,
            "shootingDiff": shooting_diff,
            "playmakingDiff": playmaking_diff,
            "defenseDiff": defense_diff,
            "reboundingDiff": rebounding_diff,
            "physicalDiff": physical_diff,
        }]
    )

    predicted_margin = (
        linear_model.predict(
            features
        )[0]
    )

    home_win_probability = (
        logistic_model.predict_proba(
            features
        )[0, 1]
    )

    return {
        "predictedMargin": float(predicted_margin),
        "homeWinProbability": float(home_win_probability),
        "predictedHomeWin": bool(
            home_win_probability >= 0.5
        ),
    }

def get_latest_team_ratings(
    team,
    game_features_df,
):
    team_games = game_features_df[
        (game_features_df["homeTeam"] == team)
        | (game_features_df["awayTeam"] == team)
    ].copy()

    if team_games.empty:
        raise ValueError(
            f"No game data found for team: {team}"
        )

    team_games = team_games.sort_values(
        "gameDate"
    )

    latest_game = team_games.iloc[-1]

    is_home = (
        latest_game["homeTeam"] == team
    )

    prefix = (
        "home"
        if is_home
        else "away"
    )

    ratings = {}

    for rating in TEAM_RATINGS:
        ratings[rating.lower()] = (
            latest_game[
                f"{prefix}{rating}"
            ]
        )

    return ratings

def predict_game(
    home_team,
    away_team,
):
    game_features_df = pd.read_csv(
        GAME_FEATURES_PATH,
        parse_dates=["gameDate"],
    )

    home_ratings = get_latest_team_ratings(
        home_team,
        game_features_df,
    )

    away_ratings = get_latest_team_ratings(
        away_team,
        game_features_df,
    )

    result = predict_from_features(
        finishing_diff=(
            home_ratings["finishing"]
            - away_ratings["finishing"]
        ),
        shooting_diff=(
            home_ratings["shooting"]
            - away_ratings["shooting"]
        ),
        playmaking_diff=(
            home_ratings["playmaking"]
            - away_ratings["playmaking"]
        ),
        defense_diff=(
            home_ratings["defense"]
            - away_ratings["defense"]
        ),
        rebounding_diff=(
            home_ratings["rebounding"]
            - away_ratings["rebounding"]
        ),
        physical_diff=(
            home_ratings["physical"]
            - away_ratings["physical"]
        ),
    )

    result["homeTeam"] = home_team
    result["awayTeam"] = away_team

    result["predictedWinner"] = (
        home_team
        if result["predictedHomeWin"]
        else away_team
    )

    return result

if __name__ == "__main__":
    result = predict_game(
        home_team="OKC",
        away_team="HOU",
    )

    print(
        f"Home Team: "
        f"{result['homeTeam']}"
    )

    print(
        f"Away Team: "
        f"{result['awayTeam']}"
    )

    print(
        f"Predicted Margin: "
        f"{result['predictedMargin']:.2f}"
    )

    print(
        f"Home Win Probability: "
        f"{result['homeWinProbability']:.1%}"
    )

    print(
        f"Predicted Winner: "
        f"{result['predictedWinner']}"
    )