from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    log_loss,
)
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LINEAR_MODEL_PATH = (
    MODEL_DIR
    / "linear_game_model_v1.joblib"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIR
    / "logistic_game_model_v1.joblib"
)

PROCESSED_DATA_DIR = Path("data/processed")

INPUT_PATH = (
    PROCESSED_DATA_DIR
    / "game_features_2025_26.csv"
)

FEATURE_COLS = [
    "finishingDiff",
    "shootingDiff",
    "playmakingDiff",
    "defenseDiff",
    "reboundingDiff",
    "physicalDiff",
]

TARGET_COL = "homePointDiff"


def load_data():
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["gameDate"],
    )

    df = df.sort_values(
        "gameDate"
    ).reset_index(drop=True)

    return df

def chronological_split(
    df,
    train_ratio=0.8,
):
    split_index = int(
        len(df) * train_ratio
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    return train_df, test_df

def train_linear_model(
    train_df,
    test_df,
):
    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        TARGET_COL
    ]

    X_test = test_df[
        FEATURE_COLS
    ]

    y_test = test_df[
        TARGET_COL
    ]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    predicted_home_win = (
        predictions > 0
    )

    actual_home_win = (
        y_test > 0
    )

    winner_accuracy = (
        predicted_home_win
        == actual_home_win
    ).mean()

    return (
        model,
        predictions,
        mae,
        rmse,
        winner_accuracy,
    )

def train_logistic_model(
    train_df,
    test_df,
):
    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        "homeWin"
    ]

    X_test = test_df[
        FEATURE_COLS
    ]

    y_test = test_df[
        "homeWin"
    ]

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    loss = log_loss(
        y_test,
        probabilities,
    )

    return (
        model,
        probabilities,
        predictions,
        accuracy,
        loss,
    )

def train_random_forest_model(
    train_df,
    test_df,
):
    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        "homeWin"
    ]

    X_test = test_df[
        FEATURE_COLS
    ]

    y_test = test_df[
        "homeWin"
    ]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    loss = log_loss(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    return (
        model,
        probabilities,
        predictions,
        accuracy,
        loss,
        brier,
    )

def train_xgboost_model(
    train_df,
    test_df,
):
    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        "homeWin"
    ]

    X_test = test_df[
        FEATURE_COLS
    ]

    y_test = test_df[
        "homeWin"
    ]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    loss = log_loss(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    return (
        model,
        probabilities,
        predictions,
        accuracy,
        loss,
        brier,
    )

if __name__ == "__main__":
    df = load_data()

    train_df, test_df = (
        chronological_split(df)
    )

    (
        model,
        predictions,
        mae,
        rmse,
        winner_accuracy,
    ) = train_linear_model(
        train_df,
        test_df,
    )

    (
        logistic_model,
        home_win_probabilities,
        logistic_predictions,
        logistic_accuracy,
        logistic_log_loss,
    ) = train_logistic_model(
        train_df,
        test_df,
    )

    print(
        "\nLogistic Regression"
    )

    print(
        "Winner accuracy:",
        logistic_accuracy,
    )

    print(
        "Log loss:",
        logistic_log_loss,
    )

    print(
        "\nLogistic coefficients:"
    )

    for feature, coef in zip(
        FEATURE_COLS,
        logistic_model.coef_[0],
    ):
        print(
            feature,
            coef,
        )

    print(
        "Intercept:",
        logistic_model.intercept_[0],
    )

    logistic_results = test_df[
        [
            "gameDate",
            "homeTeam",
            "awayTeam",
            "homeWin",
        ]
    ].copy()

    logistic_results[
        "homeWinProbability"
    ] = home_win_probabilities

    logistic_results[
        "predictedHomeWin"
    ] = logistic_predictions

    print(
        logistic_results
        .head(30)
        .to_string(index=False)
    )

    brier = brier_score_loss(
        test_df["homeWin"],
        home_win_probabilities,
    )

    print(
        "Brier score:",
        brier,
    )

    print(
        "Train games:",
        len(train_df),
    )

    print(
        "Test games:",
        len(test_df),
    )

    print(
        "Train period:",
        train_df["gameDate"].min(),
        "to",
        train_df["gameDate"].max(),
    )

    print(
        "Test period:",
        test_df["gameDate"].min(),
        "to",
        test_df["gameDate"].max(),
    )

    (
        rf_model,
        rf_probabilities,
        rf_predictions,
        rf_accuracy,
        rf_log_loss,
        rf_brier,
    ) = train_random_forest_model(
        train_df,
        test_df,
    )

    print(
        "\nRandom Forest"
    )

    print(
        "Winner accuracy:",
        rf_accuracy,
    )

    print(
        "Log loss:",
        rf_log_loss,
    )

    print(
        "Brier score:",
        rf_brier,
    )

    print(
        "\nRandom Forest feature importance:"
    )

    for feature, importance in zip(
        FEATURE_COLS,
        rf_model.feature_importances_,
    ):
        print(
            feature,
            importance,
        )

    (
        xgb_model,
        xgb_probabilities,
        xgb_predictions,
        xgb_accuracy,
        xgb_log_loss,
        xgb_brier,
    ) = train_xgboost_model(
        train_df,
        test_df,
    )

    print(
        "\nXGBoost"
    )

    print(
        "Winner accuracy:",
        xgb_accuracy,
    )

    print(
        "Log loss:",
        xgb_log_loss,
    )

    print(
        "Brier score:",
        xgb_brier,
    )

    print(
        "\nXGBoost feature importance:"
    )

    for feature, importance in zip(
        FEATURE_COLS,
        xgb_model.feature_importances_,
    ):
        print(
            feature,
            importance,
        )

    print("MAE:", mae)
    print("RMSE:", rmse)

    print(
        "Winner accuracy:",
        winner_accuracy,
    )

    print("\nCoefficients:")

    for feature, coef in zip(
        FEATURE_COLS,
        model.coef_,
    ):
        print(
            feature,
            coef,
        )

    print(
        "Intercept:",
        model.intercept_,
    )

    test_home_win_rate = (
        test_df["homePointDiff"] > 0
    ).mean()

    print(
        "Test home win baseline:",
        test_home_win_rate,
    )

    correlation = np.corrcoef(
        predictions,
        test_df["homePointDiff"],
    )[0, 1]

    print(
        "Prediction correlation:",
        correlation,
    )

    results = test_df[
        [
            "gameDate",
            "homeTeam",
            "awayTeam",
            "homePointDiff",
        ]
    ].copy()

    results["predictedPointDiff"] = (
        predictions
    )

    results["actualHomeWin"] = (
        results["homePointDiff"] > 0
    )

    results["predictedHomeWin"] = (
        results["predictedPointDiff"] > 0
    )

    print(
        results.head(30).to_string(
            index=False
        )
    )

    results["correctWinner"] = (
        results["actualHomeWin"]
        == results["predictedHomeWin"]
    )

    results["predictionConfidence"] = (
        results["predictedPointDiff"].abs()
    )

    print(
        results.groupby(
            pd.cut(
                results["predictionConfidence"],
                bins=[
                    0,
                    3,
                    6,
                    10,
                    20,
                    float("inf"),
                ],
            ),
            observed=False,
        )["correctWinner"]
        .agg(["count", "mean"])
    )

    benchmark = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
            ],
            "WinnerAccuracy": [
                winner_accuracy,
                logistic_accuracy,
                rf_accuracy,
                xgb_accuracy,
            ],
            "LogLoss": [
                np.nan,
                logistic_log_loss,
                rf_log_loss,
                xgb_log_loss,
            ],
            "BrierScore": [
                np.nan,
                brier,
                rf_brier,
                xgb_brier,
            ],
            "MAE": [
                mae,
                np.nan,
                np.nan,
                np.nan,
            ],
            "RMSE": [
                rmse,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    print("\nModel Benchmark")
    print(
        benchmark.to_string(
            index=False
        )
    )

    joblib.dump(
        model,
        LINEAR_MODEL_PATH,
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH,
    )

    print(
        f"Saved linear model to "
        f"{LINEAR_MODEL_PATH}"
    )

    print(
        f"Saved logistic model to "
        f"{LOGISTIC_MODEL_PATH}"
    )