from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)

from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/player_level_game_features.csv"
)

PLAYER_RATING_FEATURES = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
]

PLAYER_FEATURES = [
    *PLAYER_RATING_FEATURES,
    "expectedMinutesRosterAdjusted",
]

TOP_N_PLAYERS = 10

SEEDS = [
    7,
    21,
    42,
    84,
    123,
]

def build_temporal_folds(
    train_df,
):
    fold_specs = [
        {
            "fit_end": "2025-02-15",
            "validation_start": "2025-02-16",
            "validation_end": "2025-04-13",
        },
        {
            "fit_end": "2025-11-08",
            "validation_start": "2025-11-09",
            "validation_end": "2025-12-05",
        },
        {
            "fit_end": "2025-12-19",
            "validation_start": "2025-12-20",
            "validation_end": "2026-01-29",
        },
    ]

    folds = []

    for spec in fold_specs:
        fit_end = pd.Timestamp(
            spec["fit_end"]
        )

        validation_start = pd.Timestamp(
            spec["validation_start"]
        )

        validation_end = pd.Timestamp(
            spec["validation_end"]
        )

        fit_df = train_df[
            train_df["gameDate"]
            <= fit_end
        ].copy()

        validation_df = train_df[
            (
                train_df["gameDate"]
                >= validation_start
            )
            &
            (
                train_df["gameDate"]
                <= validation_end
            )
        ].copy()

        folds.append(
            (
                fit_df,
                validation_df,
            )
        )

    return folds

def get_player_feature_cols(
    df,
):
    feature_cols = []

    for team_prefix in [
        "home",
        "away",
    ]:
        for player_number in range(
            1,
            TOP_N_PLAYERS + 1,
        ):
            for feature in PLAYER_FEATURES:
                col = (
                    f"{team_prefix}"
                    f"_p{player_number}"
                    f"_{feature}"
                )

                if col not in df.columns:
                    raise ValueError(
                        f"Missing feature column: {col}"
                    )

                feature_cols.append(
                    col
                )

    return feature_cols

def evaluate_point_diff(
    y_true_diff,
    predicted_diff,
):
    predicted_home_win = (
        predicted_diff > 0
    ).astype(int)

    actual_home_win = (
        y_true_diff > 0
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            actual_home_win,
            predicted_home_win,
        ),
        "mae": mean_absolute_error(
            y_true_diff,
            predicted_diff,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true_diff,
                predicted_diff,
            )
        ),
    }

def get_point_diff(
    df,
):
    return (
        df[
            "homePointDiff"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

def build_team_feature_matrix(
    df,
):
    team_features = []

    for _, row in df.iterrows():
        team_rating_vectors = {}

        for team_prefix in [
            "home",
            "away",
        ]:
            ratings = []
            minutes = []

            for player_number in range(
                1,
                TOP_N_PLAYERS + 1,
            ):
                ratings.append(
                    [
                        row[
                            f"{team_prefix}"
                            f"_p{player_number}"
                            f"_{feature}"
                        ]
                        for feature
                        in PLAYER_RATING_FEATURES
                    ]
                )

                minutes.append(
                    row[
                        f"{team_prefix}"
                        f"_p{player_number}"
                        "_expectedMinutesRosterAdjusted"
                    ]
                )

            ratings = np.asarray(
                ratings,
                dtype=np.float32,
            )

            minutes = np.asarray(
                minutes,
                dtype=np.float32,
            )

            minute_sum = (
                minutes.sum()
            )

            if minute_sum > 0:
                weights = (
                    minutes
                    / minute_sum
                )

                team_rating = (
                    ratings
                    * weights[:, None]
                ).sum(
                    axis=0
                )

            else:
                team_rating = (
                    ratings.mean(
                        axis=0
                    )
                )

            team_rating_vectors[
                team_prefix
            ] = team_rating

        rating_diff = (
            team_rating_vectors[
                "home"
            ]
            - team_rating_vectors[
                "away"
            ]
        )

        team_features.append(
            rating_diff
        )

    return np.asarray(
        team_features,
        dtype=np.float32,
    )

def evaluate_probabilities(
    y_true,
    probabilities,
):
    predictions = (
        probabilities
        >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "logloss": log_loss(
            y_true,
            probabilities,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
    }

def train_player_linear(
    fit_df,
    validation_df,
    feature_cols,
):
    X_fit = (
        fit_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_validation = (
        validation_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_fit = get_point_diff(
        fit_df
    )

    y_validation = get_point_diff(
        validation_df
    )

    scaler = StandardScaler()

    X_fit = scaler.fit_transform(
        X_fit
    )

    X_validation = scaler.transform(
        X_validation
    )

    model = LinearRegression()

    model.fit(
        X_fit,
        y_fit,
    )

    predicted_diff = model.predict(
        X_validation
    )

    return evaluate_point_diff(
        y_validation,
        predicted_diff,
    )

def train_player_logistic(
    fit_df,
    validation_df,
    feature_cols,
):
    X_fit = (
        fit_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_validation = (
        validation_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_fit = (
        fit_df[
            "homeWin"
        ]
        .to_numpy()
    )

    y_validation = (
        validation_df[
            "homeWin"
        ]
        .to_numpy()
    )

    scaler = StandardScaler()

    X_fit = (
        scaler.fit_transform(
            X_fit
        )
    )

    X_validation = (
        scaler.transform(
            X_validation
        )
    )

    model = LogisticRegression(
        max_iter=2000,
    )

    model.fit(
        X_fit,
        y_fit,
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    return evaluate_probabilities(
        y_validation,
        probabilities,
    )

def train_player_xgboost(
    fit_df,
    validation_df,
    feature_cols,
    seed,
):
    X_fit = (
        fit_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_validation = (
        validation_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_fit = (
        fit_df[
            "homeWin"
        ]
        .to_numpy()
    )

    y_validation = (
        validation_df[
            "homeWin"
        ]
        .to_numpy()
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_fit,
        y_fit,
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    return evaluate_probabilities(
        y_validation,
        probabilities,
    )

def train_team_linear(
    fit_df,
    validation_df,
):
    X_fit = (
        build_team_feature_matrix(
            fit_df
        )
    )

    X_validation = (
        build_team_feature_matrix(
            validation_df
        )
    )

    y_fit = get_point_diff(
        fit_df
    )

    y_validation = get_point_diff(
        validation_df
    )

    scaler = StandardScaler()

    X_fit = scaler.fit_transform(
        X_fit
    )

    X_validation = scaler.transform(
        X_validation
    )

    model = LinearRegression()

    model.fit(
        X_fit,
        y_fit,
    )

    predicted_diff = model.predict(
        X_validation
    )

    return evaluate_point_diff(
        y_validation,
        predicted_diff,
    )

def train_team_logistic(
    fit_df,
    validation_df,
):
    X_fit = (
        build_team_feature_matrix(
            fit_df
        )
    )

    X_validation = (
        build_team_feature_matrix(
            validation_df
        )
    )

    y_fit = (
        fit_df[
            "homeWin"
        ]
        .to_numpy()
    )

    y_validation = (
        validation_df[
            "homeWin"
        ]
        .to_numpy()
    )

    scaler = StandardScaler()

    X_fit = scaler.fit_transform(
        X_fit
    )

    X_validation = scaler.transform(
        X_validation
    )

    model = LogisticRegression(
        max_iter=2000,
    )

    model.fit(
        X_fit,
        y_fit,
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    return evaluate_probabilities(
        y_validation,
        probabilities,
    )

def train_team_xgboost(
    fit_df,
    validation_df,
    seed,
):
    X_fit = (
        build_team_feature_matrix(
            fit_df
        )
    )

    X_validation = (
        build_team_feature_matrix(
            validation_df
        )
    )

    y_fit = (
        fit_df[
            "homeWin"
        ]
        .to_numpy()
    )

    y_validation = (
        validation_df[
            "homeWin"
        ]
        .to_numpy()
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_fit,
        y_fit,
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    return evaluate_probabilities(
        y_validation,
        probabilities,
    )

def main():
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

    feature_cols = (
        get_player_feature_cols(
            df
        )
    )
    team_test = (
        build_team_feature_matrix(
            df.head(5)
        )
    )

    print(
        "Player feature count:",
        len(feature_cols),
    )

    print(
        "Team feature shape:",
        team_test.shape,
    )

    print(
        "First team feature row:",
        team_test[0],
    )

    cutoff_date = pd.Timestamp(
        "2026-01-30"
    )

    train_df = df[
        df["gameDate"]
        < cutoff_date
    ].copy()

    temporal_folds = (
        build_temporal_folds(
            train_df
        )
    )

    results = []

    for (
        fold_number,
        (
            fit_df,
            validation_df,
        ),
    ) in enumerate(
        temporal_folds,
        start=1,
    ):
        player_linear_result = (
            train_player_linear(
                fit_df,
                validation_df,
                feature_cols,
            )
        )

        team_linear_result = (
            train_team_linear(
                fit_df,
                validation_df,
            )
        )
        
        logistic_result = (
            train_player_logistic(
                fit_df,
                validation_df,
                feature_cols,
            )
        )

        team_logistic_result = (
            train_team_logistic(
                fit_df,
                validation_df,
            )
        )

        for (
            model_name,
            result,
        ) in [
            (
                "Player PointDiff Linear",
                player_linear_result,
            ),
            (
                "Team PointDiff Linear",
                team_linear_result,
            ),
            (
                "Player Logistic",
                logistic_result,
            ),
            (
                "Team Logistic",
                team_logistic_result,
            ),
        ]:
            results.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    **result,
                }
            )

        for seed in SEEDS:
            player_xgb_result = (
                train_player_xgboost(
                    fit_df,
                    validation_df,
                    feature_cols,
                    seed,
                )
            )

            team_xgb_result = (
                train_team_xgboost(
                    fit_df,
                    validation_df,
                    seed,
                )
            )

            results.append(
                {
                    "model": "Player XGBoost",
                    "fold": fold_number,
                    "seed": seed,
                    **player_xgb_result,
                }
            )

            results.append(
                {
                    "model": "Team XGBoost",
                    "fold": fold_number,
                    "seed": seed,
                    **team_xgb_result,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    # ------------------------------------------
    # Point-differential regression results
    # ------------------------------------------

    linear_results = (
        results_df[
            results_df[
                "model"
            ].str.contains(
                "PointDiff Linear"
            )
        ]
    )

    print()
    print(
        "================================"
    )
    print(
        "Point-Differential Regression"
    )
    print(
        "================================"
    )

    print(
        linear_results
        .groupby(
            "model"
        )[
            [
                "accuracy",
                "mae",
                "rmse",
            ]
        ]
        .mean()
    )

    print()
    print(
        "Point-Diff Results by Fold:"
    )

    print(
        linear_results
        .groupby(
            [
                "model",
                "fold",
            ]
        )[
            [
                "accuracy",
                "mae",
                "rmse",
            ]
        ]
        .mean()
    )


    # ------------------------------------------
    # Win-probability classification results
    # ------------------------------------------

    classification_results = (
        results_df[
            ~results_df[
                "model"
            ].str.contains(
                "PointDiff Linear"
            )
        ]
    )

    print()
    print(
        "================================"
    )
    print(
        "Win Probability Models"
    )
    print(
        "================================"
    )

    print(
        classification_results
        .groupby(
            "model"
        )[
            [
                "accuracy",
                "logloss",
                "brier",
            ]
        ]
        .mean()
    )

    print()
    print(
        "Classification Results by Fold:"
    )

    print(
        classification_results
        .groupby(
            [
                "model",
                "fold",
            ]
        )[
            [
                "accuracy",
                "logloss",
                "brier",
            ]
        ]
        .mean()
    )
    team_matrix = (
        build_team_feature_matrix(
            train_df
        )
    )

    print()
    print(
        "Team feature means:",
        team_matrix.mean(
            axis=0
        )
    )

    print(
        "Team feature stds:",
        team_matrix.std(
            axis=0
        )
    )

    print(
        "Team feature mins:",
        team_matrix.min(
            axis=0
        )
    )

    print(
        "Team feature maxs:",
        team_matrix.max(
            axis=0
        )
    )


if __name__ == "__main__":
    main()