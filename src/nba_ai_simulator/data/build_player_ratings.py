from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")

INPUT_PATH = (
    PROCESSED_DATA_DIR
    / "player_games_two_seasons.csv"
)

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "player_ratings.csv"
)

PLAYER_PROFILES_PATH = (
    PROCESSED_DATA_DIR
    / "player_profiles.csv"
)

def convert_minutes(value):
    if pd.isna(value):
        return np.nan

    minutes, seconds = value.split(":")

    return int(minutes) + int(seconds) / 60

def prepare_player_games(player_games_df):
    df = player_games_df.copy()

    df["minutesNumeric"] = (
        df["minutes"]
        .apply(convert_minutes)
    )

    df = df[
        df["minutesNumeric"] > 0
    ].copy()

    df["twoPointersMade"] = (
        df["fieldGoalsMade"]
        - df["threePointersMade"]
    )

    df["twoPointersAttempted"] = (
        df["fieldGoalsAttempted"]
        - df["threePointersAttempted"]
    )

    return df

ADVANCED_WEIGHTED_COLS = [
    "usagePercentage",
    "assistPercentage",
    "assistToTurnover",
    "assistRatio",
    "turnoverRatio",
    "offensiveReboundPercentage",
    "defensiveReboundPercentage",
    "reboundPercentage",
    "offensiveRating",
    "defensiveRating",
    "netRating",
    "PIE",
]

def add_weighted_advanced_columns(df):
    df = df.copy()

    for col in ADVANCED_WEIGHTED_COLS:
        df[f"{col}Weighted"] = (
            df[col]
            * df["minutesNumeric"]
        )

    return df

def build_player_profiles(df):
    player_profile_df = (
    df
    .groupby(
        ["personId", "firstName", "familyName"],
        as_index=False
    )
    .agg(
        gamesPlayed=("gameId", "nunique"),
        totalMinutes=("minutesNumeric", "sum"),

        # scoring volume
        points=("points", "sum"),

        # shooting totals
        fieldGoalsMade=("fieldGoalsMade", "sum"),
        fieldGoalsAttempted=("fieldGoalsAttempted", "sum"),
        threePointersMade=("threePointersMade", "sum"),
        threePointersAttempted=("threePointersAttempted", "sum"),
        freeThrowsMade=("freeThrowsMade", "sum"),
        freeThrowsAttempted=("freeThrowsAttempted", "sum"),
        twoPointersMade=("twoPointersMade", "sum"),
        twoPointersAttempted=("twoPointersAttempted", "sum"),

        assists=("assists", "sum"),
        turnovers=("turnovers", "sum"),
        steals=("steals", "sum"),
        blocks=("blocks", "sum"),
        reboundsOffensive=("reboundsOffensive", "sum"),
        reboundsDefensive=("reboundsDefensive", "sum"),
        reboundsTotal=("reboundsTotal", "sum"),

        usagePercentageWeighted=("usagePercentageWeighted", "sum"),
        assistPercentageWeighted=("assistPercentageWeighted", "sum"),
        assistToTurnoverWeighted=("assistToTurnoverWeighted", "sum"),
        assistRatioWeighted=("assistRatioWeighted", "sum"),
        turnoverRatioWeighted=("turnoverRatioWeighted", "sum"),

        offensiveReboundPercentageWeighted=("offensiveReboundPercentageWeighted", "sum"),
        defensiveReboundPercentageWeighted=("defensiveReboundPercentageWeighted", "sum"),
        reboundPercentageWeighted=("reboundPercentageWeighted", "sum"),

        offensiveRatingWeighted=("offensiveRatingWeighted", "sum"),
        defensiveRatingWeighted=("defensiveRatingWeighted", "sum"),
        netRatingWeighted=("netRatingWeighted", "sum"),
        PIEWeighted=("PIEWeighted", "sum"),
        )
    )
    player_profile_df["twoPointPercentage"] = (
        player_profile_df["twoPointersMade"]
        / player_profile_df["twoPointersAttempted"].replace(0, np.nan)
    )

    player_profile_df["threePointersPercentage"] = (
        player_profile_df["threePointersMade"]
        / player_profile_df["threePointersAttempted"].replace(0, np.nan)
    )

    player_profile_df["fieldGoalPercentage"] = (
        player_profile_df["fieldGoalsMade"]
        / player_profile_df["fieldGoalsAttempted"].replace(0, np.nan)
    )

    player_profile_df["freeThrowPercentage"] = (
        player_profile_df["freeThrowsMade"]
        / player_profile_df["freeThrowsAttempted"].replace(0, np.nan)
    )

    player_profile_df["trueShootingPercentage"] = (
        player_profile_df["points"]
        / (
            2 * (
                player_profile_df["fieldGoalsAttempted"]
                + 0.44 * player_profile_df["freeThrowsAttempted"]
            )
        ).replace(0, np.nan)
    )

    player_profile_df["effectiveFieldGoalPercentage"] = (
        (
            player_profile_df["fieldGoalsMade"]
            + 0.5 * player_profile_df["threePointersMade"]
        )
        / player_profile_df["fieldGoalsAttempted"].replace(0, np.nan)
    )

    player_profile_df["pointsPerMinute"] = (
        player_profile_df["points"]
        / player_profile_df["totalMinutes"]
    )

    player_profile_df["assistsPerMinute"] = (
        player_profile_df["assists"]
        / player_profile_df["totalMinutes"]
    )

    player_profile_df["stealsPerMinute"] = (
        player_profile_df["steals"]
        / player_profile_df["totalMinutes"]
    )

    player_profile_df["blocksPerMinute"] = (
        player_profile_df["blocks"]
        / player_profile_df["totalMinutes"]
    )

    player_profile_df["threePointAttemptRate"] = (
        player_profile_df["threePointersAttempted"]
        / player_profile_df["fieldGoalsAttempted"].replace(0, np.nan)
    )

    player_profile_df["freeThrowRate"] = (
        player_profile_df["freeThrowsAttempted"]
        / player_profile_df["fieldGoalsAttempted"].replace(0, np.nan)
    )

    for col in ADVANCED_WEIGHTED_COLS:
        player_profile_df[col] = (
            player_profile_df[f"{col}Weighted"]
            / player_profile_df["totalMinutes"]
        )

    return player_profile_df

def percentile_score(series):
    return series.rank(pct=True) * 100

def weighted_rating(components):
    numerator = 0
    denominator = 0

    for series, weight in components:
        valid = series.notna()

        numerator += series.fillna(0) * weight
        denominator += valid.astype(float) * weight

    return numerator / denominator.replace(0, np.nan)

def build_player_ratings(player_profile_df):
    df = player_profile_df.copy()

    df["finishing"] = weighted_rating([
        (
            percentile_score(df["twoPointPercentage"]),
            0.40,
        ),
        (
            percentile_score(df["freeThrowRate"]),
            0.25,
        ),
        (
            percentile_score(df["pointsPerMinute"]),
            0.20,
        ),
        (
            percentile_score(df["trueShootingPercentage"]),
            0.15,
        ),
    ])

    df["shooting"] = weighted_rating([
        (
            percentile_score(df["threePointersPercentage"]),
            0.40,
        ),
        (
            percentile_score(df["threePointAttemptRate"]),
            0.25,
        ),
        (
            percentile_score(df["trueShootingPercentage"]),
            0.20,
        ),
        (
            percentile_score(df["freeThrowPercentage"]),
            0.15,
        ),
    ])

    df["playmaking"] = weighted_rating([
        (
            percentile_score(df["assistPercentage"]),
            0.40,
        ),
        (
            percentile_score(df["assistToTurnover"]),
            0.25,
        ),
        (
            percentile_score(df["assistRatio"]),
            0.20,
        ),
        (
            100 - percentile_score(df["turnoverRatio"]),
            0.15,
        ),
    ])

    df["rebounding"] = weighted_rating([
        (
            percentile_score(df["reboundPercentage"]),
            0.45,
        ),
        (
            percentile_score(
                df["offensiveReboundPercentage"]
            ),
            0.30,
        ),
        (
            percentile_score(
                df["defensiveReboundPercentage"]
            ),
            0.25,
        ),
    ])

    df["defense"] = weighted_rating([
        (
            percentile_score(df["stealsPerMinute"]),
            0.30,
        ),
        (
            percentile_score(df["blocksPerMinute"]),
            0.30,
        ),
        (
            100 - percentile_score(df["defensiveRating"]),
            0.40,
        ),
    ])

    return df

def map_physical_group(position):
    if position == "Guard":
        return "Guard"

    if position in [
        "Guard-Forward",
        "Forward-Guard",
    ]:
        return "Wing"

    if position == "Forward":
        return "Forward"

    if position in [
        "Forward-Center",
        "Center-Forward",
        "Center",
    ]:
        return "Big"

    return "Unknown"

def add_physical_rating(ratings_df, profiles_df):
    profiles_df = profiles_df.copy()

    profiles_df["physicalGroup"] = (
        profiles_df["position"]
        .apply(map_physical_group)
    )

    profiles_df["weight"] = (
        profiles_df["weight"]
        .fillna(
            profiles_df.groupby(
                "physicalGroup"
            )["weight"].transform("median")
        )
    )

    profiles_df["heightPercentile"] = (
        profiles_df
        .groupby("physicalGroup")["heightInches"]
        .rank(pct=True)
        * 100
    )

    profiles_df["weightPercentile"] = (
        profiles_df
        .groupby("physicalGroup")["weight"]
        .rank(pct=True)
        * 100
    )

    profiles_df["physical"] = (
        0.55 * profiles_df["heightPercentile"]
        + 0.45 * profiles_df["weightPercentile"]
    )

    ratings_df = ratings_df.merge(
        profiles_df[
            [
                "personId",
                "position",
                "physicalGroup",
                "heightInches",
                "weight",
                "birthDate",
                "physical",
            ]
        ],
        on="personId",
        how="left",
        validate="one_to_one",
    )
    
    return ratings_df

def main():
    player_games_df = pd.read_csv(
        INPUT_PATH,
        dtype={"gameId": str},
    )

    prepared_df = prepare_player_games(
        player_games_df
    )

    prepared_df = add_weighted_advanced_columns(
        prepared_df
    )

    player_profile_df = build_player_profiles(
        prepared_df
    )

    ratings_df = build_player_ratings(
        player_profile_df
    )

    profiles_df = pd.read_csv(
        PLAYER_PROFILES_PATH
    )

    ratings_df = add_physical_rating(
        ratings_df,
        profiles_df,
    )


    rating_cols = [
        "finishing",
        "shooting",
        "playmaking",
        "defense",
        "rebounding",
        "physical",
    ]

    ratings_df[rating_cols] = (
        ratings_df[rating_cols]
        .round(1)
    )

    ratings_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(ratings_df.shape)
    print(ratings_df[rating_cols].isna().sum())
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
