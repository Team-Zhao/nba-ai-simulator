import numpy as np
import pandas as pd
from pathlib import Path

RECENT_GAMES = 10

PROCESSED_DATA_DIR = Path("data/processed")

PLAYER_RATINGS_PATH = (
    PROCESSED_DATA_DIR
    / "player_ratings_train.csv"
)

PLAYER_RATINGS_PATH = (
    PROCESSED_DATA_DIR
    / "player_ratings.csv"
)

RATING_COLS = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
]

def merge_player_ratings(
    player_games_df,
    ratings_df,
):
    rating_features = [
        "personId",
        *RATING_COLS,
    ]

    df = player_games_df.merge(
        ratings_df[rating_features],
        on="personId",
        how="left",
        validate="many_to_one",
    )

    return df

def convert_minutes(value):
    if pd.isna(value):
        return np.nan

    minutes, seconds = value.split(":")

    return int(minutes) + int(seconds) / 60

def add_minutes_numeric(player_games_df):
    df = player_games_df.copy()

    df["minutesNumeric"] = (
        df["minutes"]
        .apply(convert_minutes)
    )

    return df

def add_expected_minutes(player_games_df):
    df = player_games_df.copy()

    df["gameDate"] = pd.to_datetime(
        df["gameDate"]
    )

    df = df.sort_values(
        ["personId", "gameDate"]
    )

    df = df.sort_values(
        ["personId", "gameDate"]
    )

    df["minutesForHistory"] = (
        df["minutesNumeric"]
        .fillna(0)
    )

    df["expectedMinutes"] = (
        df
        .groupby("personId")["minutesForHistory"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    window=RECENT_GAMES,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    return df

def add_expected_minutes_fallback(df):
    df = df.copy()

    # Current game's starting lineup is known before tip-off.
    df["isStarter"] = df["position"].notna()

    df["expectedMinutes"] = df["expectedMinutes"].fillna(
        df["isStarter"].map({
            True: 30.0,
            False: 12.0,
        })
    )

    return df

def prepare_team_feature_data():
    player_games_df = pd.read_csv(
        PLAYER_GAMES_PATH,
        dtype={"gameId": str},
    )

    ratings_df = pd.read_csv(
        PLAYER_RATINGS_PATH
    )

    df = add_minutes_numeric(
        player_games_df
    )

    df = add_expected_minutes(
        df
    )

    df = add_expected_minutes_fallback(
        df
    )

    df = merge_player_ratings(
        df,
        ratings_df,
    )

    return df

if __name__ == "__main__":
    df = prepare_team_feature_data()

    print(df.shape)

    print(
        df[
            [
                "gameId",
                "teamTricode",
                "firstName",
                "familyName",
                "isStarter",
                "expectedMinutes",
                "finishing",
                "shooting",
                "playmaking",
                "defense",
                "rebounding",
                "physical",
            ]
        ].head(20)
    )

    print(
        df[RATING_COLS]
        .isna()
        .sum()
    )