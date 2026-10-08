from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)

from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/player_level_game_features.csv"
)

MODEL_DIR = Path(
    "models"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIR
    / "logistic_player_level_model.joblib"
)

XGBOOST_MODEL_PATH = (
    MODEL_DIR
    / "xgboost_player_level_model.joblib"
)


def chronological_split(
    df,
    train_ratio=0.8,
):
    unique_dates = (
        df["gameDate"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(unique_dates)
        * train_ratio
    )

    cutoff_date = (
        unique_dates.iloc[
            split_index
        ]
    )

    train_df = df[
        df["gameDate"]
        < cutoff_date
    ].copy()

    test_df = df[
        df["gameDate"]
        >= cutoff_date
    ].copy()

    return (
        train_df,
        test_df,
        cutoff_date,
    )


def get_player_feature_cols(
    df,
):
    feature_cols = [
        col
        for col in df.columns
        if (
            col.startswith(
                "home_p"
            )
            or col.startswith(
                "away_p"
            )
        )
    ]

    return feature_cols


def evaluate_classifier(
    name,
    model,
    X_test,
    y_test,
):
    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= 0.5
    ).astype(int)

    accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    loss = (
        log_loss(
            y_test,
            probabilities,
        )
    )

    brier = (
        brier_score_loss(
            y_test,
            probabilities,
        )
    )

    print()
    print(
        f"{name}:"
    )

    print(
        "Accuracy:",
        accuracy,
    )

    print(
        "Log loss:",
        loss,
    )

    print(
        "Brier:",
        brier,
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "logLoss": loss,
        "brier": brier,
    }


def train_logistic_model(
    X_train,
    y_train,
):
    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=5000,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_xgboost_model(
    X_train,
    y_train,
):
    model = (
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=(
                "binary:logistic"
            ),
            eval_metric="logloss",
            random_state=42,
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def main():
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "gameId": str,
        },
        parse_dates=[
            "gameDate"
        ],
    )

    df["gameId"] = (
        df["gameId"]
        .str.zfill(10)
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "Date range:",
        df["gameDate"].min(),
        "to",
        df["gameDate"].max(),
    )

    print(
        "Unique games:",
        df["gameId"].nunique(),
    )

    # --------------------------------------------------
    # Feature columns
    # --------------------------------------------------

    feature_cols = (
        get_player_feature_cols(
            df
        )
    )

    print(
        "Player-level feature count:",
        len(feature_cols),
    )

    assert (
        len(feature_cols)
        == 140
    )

    assert (
        df[feature_cols]
        .notna()
        .all()
        .all()
    )

    # --------------------------------------------------
    # Chronological split
    # --------------------------------------------------

    (
        train_df,
        test_df,
        cutoff_date,
    ) = chronological_split(
        df
    )

    print()
    print(
        "Cutoff date:",
        cutoff_date,
    )

    print(
        "Train games:",
        len(train_df),
    )

    print(
        "Train date range:",
        train_df[
            "gameDate"
        ].min(),
        "to",
        train_df[
            "gameDate"
        ].max(),
    )

    print(
        "Test games:",
        len(test_df),
    )

    print(
        "Test date range:",
        test_df[
            "gameDate"
        ].min(),
        "to",
        test_df[
            "gameDate"
        ].max(),
    )

    # --------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------

    X_train = (
        train_df[
            feature_cols
        ]
    )

    y_train = (
        train_df[
            "homeWin"
        ]
    )

    X_test = (
        test_df[
            feature_cols
        ]
    )

    y_test = (
        test_df[
            "homeWin"
        ]
    )

    print()
    print(
        "Train home win rate:",
        y_train.mean(),
    )

    print(
        "Test home win rate:",
        y_test.mean(),
    )

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    logistic_model = (
        train_logistic_model(
            X_train,
            y_train,
        )
    )

    logistic_results = (
        evaluate_classifier(
            name=(
                "Player-level Logistic"
            ),
            model=logistic_model,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    xgboost_model = (
        train_xgboost_model(
            X_train,
            y_train,
        )
    )

    xgboost_results = (
        evaluate_classifier(
            name=(
                "Player-level XGBoost"
            ),
            model=xgboost_model,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # --------------------------------------------------
    # Compare against team-level Logistic baseline
    # --------------------------------------------------

    team_logistic_accuracy = (
        0.7325581395
    )

    print()
    print(
        "Team-level Logistic baseline:",
        team_logistic_accuracy,
    )

    print(
        "Player Logistic difference:",
        logistic_results[
            "accuracy"
        ]
        - team_logistic_accuracy,
    )

    print(
        "Player XGBoost difference:",
        xgboost_results[
            "accuracy"
        ]
        - team_logistic_accuracy,
    )

    # --------------------------------------------------
    # Save models
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH,
    )

    joblib.dump(
        xgboost_model,
        XGBOOST_MODEL_PATH,
    )

    print()
    print(
        "Saved:",
        LOGISTIC_MODEL_PATH,
    )

    print(
        "Saved:",
        XGBOOST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()