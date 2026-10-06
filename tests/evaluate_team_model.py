import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)


GAME_FEATURES_PATH = (
    "data/processed/"
    "game_features_rolling.csv"
)

MODEL_PATH = (
    "models/"
    "logistic_game_model_v2.joblib"
)

FEATURE_COLS = [
    "finishingDiff",
    "shootingDiff",
    "playmakingDiff",
    "defenseDiff",
    "reboundingDiff",
    "physicalDiff",
]


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

    train_df = (
        df[
            df["gameDate"]
            < cutoff_date
        ]
        .copy()
    )

    test_df = (
        df[
            df["gameDate"]
            >= cutoff_date
        ]
        .copy()
    )

    return (
        train_df,
        test_df,
    )


def evaluate_subset(
    df,
    model,
    label,
):
    X = df[
        FEATURE_COLS
    ]

    y = (
        df["homeWin"]
        .astype(int)
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    predictions = (
        probabilities
        >= 0.5
    ).astype(int)

    accuracy = (
        accuracy_score(
            y,
            predictions,
        )
    )

    loss = (
        log_loss(
            y,
            probabilities,
        )
    )

    brier = (
        brier_score_loss(
            y,
            probabilities,
        )
    )

    print()
    print(label)
    print(
        "Games:",
        len(df),
    )
    print(
        "Date range:",
        df["gameDate"].min(),
        "to",
        df["gameDate"].max(),
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
        "Brier score:",
        brier,
    )

    return {
        "label": label,
        "games": len(df),
        "accuracy": accuracy,
        "logLoss": loss,
        "brierScore": brier,
    }


def calibration_table(
    test_df,
    model,
):
    X = test_df[
        FEATURE_COLS
    ]

    y = (
        test_df["homeWin"]
        .astype(int)
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    calibration_df = pd.DataFrame(
        {
            "homeWinProbability": (
                probabilities
            ),
            "actualHomeWin": y,
        }
    )

    calibration_df[
        "confidence"
    ] = (
        calibration_df[
            "homeWinProbability"
        ]
        .where(
            calibration_df[
                "homeWinProbability"
            ]
            >= 0.5,
            1
            - calibration_df[
                "homeWinProbability"
            ],
        )
    )

    calibration_df[
        "predictedCorrect"
    ] = (
        (
            calibration_df[
                "homeWinProbability"
            ]
            >= 0.5
        )
        == (
            calibration_df[
                "actualHomeWin"
            ]
            == 1
        )
    )

    bins = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    labels = [
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    calibration_df[
        "confidenceBin"
    ] = pd.cut(
        calibration_df[
            "confidence"
        ],
        bins=bins,
        labels=labels,
        right=False,
    )

    summary = (
        calibration_df
        .groupby(
            "confidenceBin",
            observed=False,
        )
        .agg(
            games=(
                "predictedCorrect",
                "size",
            ),
            meanConfidence=(
                "confidence",
                "mean",
            ),
            actualAccuracy=(
                "predictedCorrect",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def main():
    df = pd.read_csv(
        GAME_FEATURES_PATH,
        parse_dates=[
            "gameDate",
        ],
        dtype={
            "gameId": str,
        },
    )
    model = joblib.load(
        MODEL_PATH
    )

    _, test_df = (
        chronological_split(
            df
        )
    )

    print(
        "Full holdout:"
    )
    print(
        "Games:",
        len(test_df),
    )
    print(
        "Date range:",
        test_df[
            "gameDate"
        ].min(),
        "to",
        test_df[
            "gameDate"
        ].max(),
    )

    # ----------------------------------------
    # Full holdout metrics
    # ----------------------------------------

    full_metrics = (
        evaluate_subset(
            test_df,
            model,
            "FULL HOLDOUT",
        )
    )

    # ----------------------------------------
    # Early vs late split
    # ----------------------------------------

    unique_test_dates = (
        test_df[
            "gameDate"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    midpoint = (
        len(
            unique_test_dates
        )
        // 2
    )

    late_start_date = (
        unique_test_dates.iloc[
            midpoint
        ]
    )

    early_df = (
        test_df[
            test_df[
                "gameDate"
            ]
            < late_start_date
        ]
        .copy()
    )

    late_df = (
        test_df[
            test_df[
                "gameDate"
            ]
            >= late_start_date
        ]
        .copy()
    )

    early_metrics = (
        evaluate_subset(
            early_df,
            model,
            "EARLY HOLDOUT",
        )
    )

    late_metrics = (
        evaluate_subset(
            late_df,
            model,
            "LATE HOLDOUT",
        )
    )

    # ----------------------------------------
    # Calibration
    # ----------------------------------------

    calibration = (
        calibration_table(
            test_df,
            model,
        )
    )

    print()
    print(
        "CALIBRATION BY CONFIDENCE"
    )

    print(
        calibration.to_string(
            index=False
        )
    )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    stability_df = pd.DataFrame(
        [
            full_metrics,
            early_metrics,
            late_metrics,
        ]
    )

    stability_df.to_csv(
        "data/processed/"
        "team_model_stability.csv",
        index=False,
    )

    calibration.to_csv(
        "data/processed/"
        "team_model_calibration.csv",
        index=False,
    )

    print()
    print(
        "Saved:"
    )
    print(
        "data/processed/"
        "team_model_stability.csv"
    )
    print(
        "data/processed/"
        "team_model_calibration.csv"
    )


if __name__ == "__main__":
    main()