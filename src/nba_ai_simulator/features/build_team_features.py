import numpy as np
import pandas as pd
from pathlib import Path

RECENT_GAMES = 10

PROCESSED_DATA_DIR = Path("data/processed")

PLAYER_GAMES_PATH = (
    PROCESSED_DATA_DIR
    / "player_games_2025_26.csv"
)

PLAYER_RATINGS_PATH = (
    PROCESSED_DATA_DIR
    / "player_ratings_train.csv"
)

PLAYER_PROFILES_PATH = (
    PROCESSED_DATA_DIR
    / "player_profiles.csv"
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

    profiles_df = pd.read_csv(
        PLAYER_PROFILES_PATH
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

    df = add_cold_start_ratings(
        df,
        profiles_df,
    )

    df = add_cold_start_physical(
        df,
        profiles_df,
    )

    return df

def add_cold_start_ratings(
    df,
    profiles_df,
):
    df = df.copy()

    skill_cols = [
        "finishing",
        "shooting",
        "playmaking",
        "defense",
        "rebounding",
    ]

    df["isColdStart"] = (
        df[RATING_COLS]
        .isna()
        .all(axis=1)
    )

    for col in skill_cols:
        df.loc[
            df["isColdStart"],
            col,
        ] = 50.0

    return df

def map_physical_group(position):
    mapping = {
        "Guard": "Guard",
        "Guard-Forward": "Wing",
        "Forward-Guard": "Wing",
        "Forward": "Forward",
        "Forward-Center": "Big",
        "Center-Forward": "Big",
        "Center": "Big",
    }

    return mapping.get(position)

def add_cold_start_physical(
    df,
    profiles_df,
):
    df = df.copy()

    profile_features = profiles_df[
        [
            "personId",
            "position",
            "heightInches",
            "weight",
        ]
    ].copy()

    profile_features["physicalGroup"] = (
        profile_features["position"]
        .apply(map_physical_group)
    )

    profile_features["weight"] = (
        profile_features
        .groupby("physicalGroup")["weight"]
        .transform(
            lambda s: s.fillna(s.median())
        )
    )

    profile_features["heightScore"] = (
        profile_features
        .groupby("physicalGroup")["heightInches"]
        .rank(pct=True)
        * 100
    )

    profile_features["weightScore"] = (
        profile_features
        .groupby("physicalGroup")["weight"]
        .rank(pct=True)
        * 100
    )

    profile_features["coldStartPhysical"] = (
        0.55 * profile_features["heightScore"]
        + 0.45 * profile_features["weightScore"]
    )

    df = df.merge(
        profile_features[
            [
                "personId",
                "coldStartPhysical",
            ]
        ],
        on="personId",
        how="left",
        validate="many_to_one",
    )

    df.loc[
        df["isColdStart"],
        "physical",
    ] = df.loc[
        df["isColdStart"],
        "coldStartPhysical",
    ]

    df = df.drop(
        columns=["coldStartPhysical"]
    )

    df["physical"] = (
        df["physical"]
        .fillna(50.0)
    )

    return df

def build_team_game_ratings(df):
    df = df.copy()

    weighted_cols = []

    for rating_col in RATING_COLS:
        weighted_col = f"{rating_col}Weighted"

        df[weighted_col] = (
            df[rating_col]
            * df["expectedMinutes"]
        )

        weighted_cols.append(weighted_col)

    team_df = (
        df.groupby(
            [
                "gameId",
                "teamTricode",
            ],
            as_index=False,
        )
        .agg(
            expectedMinutesTotal=(
                "expectedMinutes",
                "sum",
            ),
            **{
                weighted_col: (
                    weighted_col,
                    "sum",
                )
                for weighted_col in weighted_cols
            },
        )
    )

    for rating_col in RATING_COLS:
        weighted_col = f"{rating_col}Weighted"

        team_df[f"team{rating_col.capitalize()}"] = (
            team_df[weighted_col]
            / team_df["expectedMinutesTotal"]
        )

    team_rating_cols = [
        f"team{rating_col.capitalize()}"
        for rating_col in RATING_COLS
    ]

    return team_df[
        [
            "gameId",
            "teamTricode",
            "expectedMinutesTotal",
            *team_rating_cols,
        ]
    ]

# if __name__ == "__main__":
#     df = prepare_team_feature_data()

#     print(df.shape)

#     print(
#         df[
#             [
#                 "gameId",
#                 "teamTricode",
#                 "firstName",
#                 "familyName",
#                 "isStarter",
#                 "expectedMinutes",
#                 "finishing",
#                 "shooting",
#                 "playmaking",
#                 "defense",
#                 "rebounding",
#                 "physical",
#             ]
#         ].head(20)
#     )

#     print(
#         df[RATING_COLS]
#         .isna()
#         .sum()
#     )

if __name__ == "__main__":
    df = prepare_team_feature_data()

    team_df = build_team_game_ratings(
        df
    )

    print(team_df.shape)
    print(team_df.head(20))

    print(
        team_df[
            [
                "teamFinishing",
                "teamShooting",
                "teamPlaymaking",
                "teamDefense",
                "teamRebounding",
                "teamPhysical",
            ]
        ]
        .isna()
        .sum()
    )

    print(
        "Games:",
        team_df["gameId"].nunique(),
    )

    print(
        "Teams per game:"
    )

    print(
        team_df.groupby("gameId")
        .size()
        .value_counts()
        .sort_index()
    )