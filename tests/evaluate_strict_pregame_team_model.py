from pathlib import Path

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

TOP_N_PLAYERS = 10

RATING_COLS = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
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


def build_team_rating(
    row,
    prefix,
    rating,
):
    weighted_total = 0.0
    minutes_total = 0.0

    for slot in range(
        1,
        TOP_N_PLAYERS + 1,
    ):
        minutes_col = (
            f"{prefix}_p{slot}_"
            "expectedMinutesRosterAdjusted"
        )

        rating_col = (
            f"{prefix}_p{slot}_{rating}"
        )

        minutes = row[
            minutes_col
        ]

        value = row[
            rating_col
        ]

        weighted_total += (
            value * minutes
        )

        minutes_total += minutes

    if minutes_total <= 0:
        raise ValueError(
            f"Non-positive minutes total "
            f"for {prefix}"
        )

    return (
        weighted_total
        / minutes_total
    )


def build_strict_team_features(
    df,
):
    output_rows = []

    for _, row in df.iterrows():
        output = {
            "gameId": row["gameId"],
            "gameDate": row["gameDate"],
            "homeTeam": row["homeTeam"],
            "awayTeam": row["awayTeam"],
            "homePointDiff": (
                row["homePointDiff"]
            ),
            "homeWin": row["homeWin"],
        }

        for rating in RATING_COLS:
            home_rating = (
                build_team_rating(
                    row=row,
                    prefix="home",
                    rating=rating,
                )
            )

            away_rating = (
                build_team_rating(
                    row=row,
                    prefix="away",
                    rating=rating,
                )
            )

            output[
                f"home{rating.capitalize()}"
            ] = home_rating

            output[
                f"away{rating.capitalize()}"
            ] = away_rating

            output[
                f"{rating}Diff"
            ] = (
                home_rating
                - away_rating
            )

        output_rows.append(
            output
        )

    return pd.DataFrame(
        output_rows
    )


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
        "accuracy": accuracy,
        "logLoss": loss,
        "brier": brier,
    }


def main():
    # --------------------------------------------------
    # Load strictly-prior player dataset
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
        "Player dataset shape:",
        df.shape,
    )

    print(
        "Date range:",
        df["gameDate"].min(),
        "to",
        df["gameDate"].max(),
    )

    # --------------------------------------------------
    # Aggregate the SAME top-10 players
    # --------------------------------------------------

    team_df = (
        build_strict_team_features(
            df
        )
    )

    print()
    print(
        "Strict team dataset shape:",
        team_df.shape,
    )

    print(
        "Unique games:",
        team_df[
            "gameId"
        ].nunique(),
    )

    assert len(
        team_df
    ) == len(
        df
    )

    assert (
        team_df[
            "gameId"
        ]
        .is_unique
    )

    # --------------------------------------------------
    # Model features
    # --------------------------------------------------

    feature_cols = [
        f"{rating}Diff"
        for rating
        in RATING_COLS
    ]

    print(
        "Strict team feature columns:",
        feature_cols,
    )

    assert (
        team_df[
            feature_cols
        ]
        .notna()
        .all()
        .all()
    )

    # --------------------------------------------------
    # Same chronological split
    # --------------------------------------------------

    (
        train_df,
        test_df,
        cutoff_date,
    ) = chronological_split(
        team_df
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

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    logistic_model = (
        LogisticRegression(
            max_iter=5000,
        )
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_results = (
        evaluate_classifier(
            name=(
                "Strict-pregame "
                "Team Logistic"
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

    xgboost_model.fit(
        X_train,
        y_train,
    )

    xgboost_results = (
        evaluate_classifier(
            name=(
                "Strict-pregame "
                "Team XGBoost"
            ),
            model=xgboost_model,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    old_team_accuracy = (
        0.7325581395
    )

    player_logistic_accuracy = (
        0.6569767441860465
    )

    player_xgboost_accuracy = (
        0.6627906976744186
    )

    print()
    print(
        "=============================="
    )

    print(
        "Representation comparison"
    )

    print(
        "=============================="
    )

    print(
        "Old rolling Team Logistic:",
        old_team_accuracy,
    )

    print(
        "Strict Team Logistic:",
        logistic_results[
            "accuracy"
        ],
    )

    print(
        "Player Logistic:",
        player_logistic_accuracy,
    )

    print()

    print(
        "Strict Team XGBoost:",
        xgboost_results[
            "accuracy"
        ],
    )

    print(
        "Player XGBoost:",
        player_xgboost_accuracy,
    )

    print()

    print(
        "Player vs strict-team "
        "Logistic difference:",
        player_logistic_accuracy
        - logistic_results[
            "accuracy"
        ],
    )

    print(
        "Player vs strict-team "
        "XGBoost difference:",
        player_xgboost_accuracy
        - xgboost_results[
            "accuracy"
        ],
    )


if __name__ == "__main__":
    main()