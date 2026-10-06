from pathlib import Path
import joblib


MODEL_DIR = Path("models")

LINEAR_MODEL_PATH = (
    MODEL_DIR
    / "linear_game_model_v2.joblib"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIR
    / "logistic_game_model_v2.joblib"
)

XGBOOST_MODEL_PATH = (
    MODEL_DIR
    / "xgboost_game_model_v2.joblib"
)

FEATURE_COLS = [
    "finishingDiff",
    "shootingDiff",
    "playmakingDiff",
    "defenseDiff",
    "reboundingDiff",
    "physicalDiff",
]

linear_model = joblib.load(
    LINEAR_MODEL_PATH
)

logistic_model = joblib.load(
    LOGISTIC_MODEL_PATH
)

xgboost_model = joblib.load(
    XGBOOST_MODEL_PATH
)

def predict_from_game_features(game_df):
    if len(game_df) != 1:
        raise ValueError(
            "Expected exactly one game row."
        )

    missing_cols = [
        col
        for col in FEATURE_COLS
        if col not in game_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing model features: "
            f"{missing_cols}"
        )

    features = game_df[
        FEATURE_COLS
    ].copy()

    predicted_margin = (
        linear_model.predict(
            features
        )[0]
    )

    logistic_home_win_probability = (
        logistic_model.predict_proba(
            features
        )[0, 1]
    )

    xgboost_home_win_probability = (
        xgboost_model.predict_proba(
            features
        )[0, 1]
    )

    home_team = (
        game_df.iloc[0]["homeTeam"]
    )

    away_team = (
        game_df.iloc[0]["awayTeam"]
    )

    predicted_home_win = (
        logistic_home_win_probability
        >= 0.5
    )

    predicted_winner = (
        home_team
        if predicted_home_win
        else away_team
    )

    return {
        "homeTeam": home_team,
        "awayTeam": away_team,
        "predictedMargin": float(
            predicted_margin
        ),
        "logisticHomeWinProbability": float(
            logistic_home_win_probability
        ),
        "xgboostHomeWinProbability": float(
            xgboost_home_win_probability
        ),
        "predictedHomeWin": bool(
            predicted_home_win
        ),
        "predictedWinner": predicted_winner,
    }
