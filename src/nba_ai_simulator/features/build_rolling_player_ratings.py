from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path("data/processed")
    / "player_games_two_seasons.csv"
)

COUNTING_STATS = [
    "minutesNumeric",
    "points",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "threePointersAttempted",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "reboundsOffensive",
    "reboundsDefensive",
    "reboundsTotal",
]

ADVANCED_METRICS = [
    "assistPercentage",
    "assistRatio",
    "turnoverRatio",
    "reboundPercentage",
    "offensiveReboundPercentage",
    "defensiveReboundPercentage",
    "defensiveRating",
]

RATING_METRICS = [
    "prior2PPercent",
    "priorFTRate",
    "priorPointsPerMinute",
    "priorTSPercent",
    "prior3PPercent",
    "prior3PARate",
    "priorFTPercent",
    "prior_assistPercentage",
    "priorASTTORatio",
    "prior_assistRatio",
    "prior_turnoverRatio",
    "prior_reboundPercentage",
    "prior_offensiveReboundPercentage",
    "prior_defensiveReboundPercentage",
    "priorStealsPerMinute",
    "priorBlocksPerMinute",
    "prior_defensiveRating",
]

def load_player_games():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["gameDate"],
    )

    df = df.sort_values(
        ["personId", "gameDate"]
    ).reset_index(drop=True)

    return df

def convert_minutes_to_numeric(value):
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value)

    if ":" in value:
        minutes, seconds = value.split(":")
        return float(minutes) + float(seconds) / 60

    return float(value)


def add_minutes_numeric(df):
    df = df.copy()

    df["minutesNumeric"] = (
        df["minutes"]
        .apply(convert_minutes_to_numeric)
    )

    return df

def add_prior_game_counts(df):
    df = df.copy()

    df["priorGames"] = (
        df.groupby("personId")
        .cumcount()
    )

    return df

def add_prior_points_average(df):
    df = df.copy()

    df["priorPointsAvg"] = (
        df.groupby("personId")["points"]
        .transform(
            lambda s: s.shift(1).expanding().mean()
        )
    )

    return df

def add_prior_cumulative_stats(df):
    df = df.copy()

    for col in COUNTING_STATS:
        df[f"prior_{col}"] = (
            df.groupby("personId")[col]
            .transform(
                lambda s: s.shift(1).cumsum()
            )
        )

    return df

def safe_divide(numerator, denominator):
    return numerator.div(
        denominator.replace(0, pd.NA)
    )

def add_prior_efficiency_metrics(df):
    df = df.copy()

    prior_2pm = (
        df["prior_fieldGoalsMade"]
        - df["prior_threePointersMade"]
    )

    prior_2pa = (
        df["prior_fieldGoalsAttempted"]
        - df["prior_threePointersAttempted"]
    )

    df["prior2PPercent"] = safe_divide(
        prior_2pm,
        prior_2pa,
    )

    df["prior3PPercent"] = safe_divide(
        df["prior_threePointersMade"],
        df["prior_threePointersAttempted"],
    )

    df["priorFTPercent"] = safe_divide(
        df["prior_freeThrowsMade"],
        df["prior_freeThrowsAttempted"],
    )

    return df

def add_prior_derived_metrics(df):
    df = df.copy()

    prior_minutes = df["prior_minutesNumeric"]

    df["priorPointsPerMinute"] = safe_divide(
        df["prior_points"],
        prior_minutes,
    )

    df["prior3PARate"] = safe_divide(
        df["prior_threePointersAttempted"],
        df["prior_fieldGoalsAttempted"],
    )

    df["priorFTRate"] = safe_divide(
        df["prior_freeThrowsAttempted"],
        df["prior_fieldGoalsAttempted"],
    )

    df["priorASTTORatio"] = safe_divide(
        df["prior_assists"],
        df["prior_turnovers"],
    )

    df["priorStealsPerMinute"] = safe_divide(
        df["prior_steals"],
        prior_minutes,
    )

    df["priorBlocksPerMinute"] = safe_divide(
        df["prior_blocks"],
        prior_minutes,
    )

    df["priorTSPercent"] = safe_divide(
        df["prior_points"],
        2 * (
            df["prior_fieldGoalsAttempted"]
            + 0.44 * df["prior_freeThrowsAttempted"]
        ),
    )

    df["priorEFGPercent"] = safe_divide(
        df["prior_fieldGoalsMade"]
        + 0.5 * df["prior_threePointersMade"],
        df["prior_fieldGoalsAttempted"],
    )

    return df

def add_prior_weighted_average_metrics(df):
    df = df.copy()

    for col in ADVANCED_METRICS:
        weighted_col = f"{col}_weighted"

        df[weighted_col] = (
            df[col] * df["minutesNumeric"]
        )

        prior_weighted_sum = (
            df.groupby("personId")[weighted_col]
            .transform(
                lambda s: s.shift(1).cumsum()
            )
        )

        df[f"prior_{col}"] = safe_divide(
            prior_weighted_sum,
            df["prior_minutesNumeric"],
        )

    return df

LOWER_IS_BETTER = {
    "prior_turnoverRatio",
    "prior_defensiveRating",
}

def add_daily_percentiles(df, metric_cols):
    df = df.copy()

    for col in metric_cols:
        ascending = col not in LOWER_IS_BETTER

        df[f"{col}_pct"] = (
            df.groupby("gameDate")[col]
            .rank(
                pct=True,
                ascending=ascending,
            )
            * 100
        )

    return df

def weighted_rating(df, components):
    weighted_sum = pd.Series(
        0.0,
        index=df.index,
    )

    weight_sum = pd.Series(
        0.0,
        index=df.index,
    )

    for col, weight in components.items():
        valid = df[col].notna()

        weighted_sum.loc[valid] += (
            df.loc[valid, col] * weight
        )

        weight_sum.loc[valid] += weight

    return safe_divide(
        weighted_sum,
        weight_sum,
    )

def add_skill_ratings(df):
    df = df.copy()

    df["finishing"] = weighted_rating(
        df,
        {
            "prior2PPercent_pct": 0.40,
            "priorFTRate_pct": 0.25,
            "priorPointsPerMinute_pct": 0.20,
            "priorTSPercent_pct": 0.15,
        },
    )

    df["shooting"] = weighted_rating(
        df,
        {
            "prior3PPercent_pct": 0.40,
            "prior3PARate_pct": 0.25,
            "priorTSPercent_pct": 0.20,
            "priorFTPercent_pct": 0.15,
        },
    )

    df["playmaking"] = weighted_rating(
        df,
        {
            "prior_assistPercentage_pct": 0.40,
            "priorASTTORatio_pct": 0.25,
            "prior_assistRatio_pct": 0.20,
            "prior_turnoverRatio_pct": 0.15,
        },
    )

    df["rebounding"] = weighted_rating(
        df,
        {
            "prior_reboundPercentage_pct": 0.45,
            "prior_offensiveReboundPercentage_pct": 0.30,
            "prior_defensiveReboundPercentage_pct": 0.25,
        },
    )

    df["defense"] = weighted_rating(
        df,
        {
            "priorStealsPerMinute_pct": 0.30,
            "priorBlocksPerMinute_pct": 0.30,
            "prior_defensiveRating_pct": 0.40,
        },
    )

    return df

def main():
    df = load_player_games()
    df = add_minutes_numeric(df)

    df = add_prior_game_counts(df)
    df = add_prior_points_average(df)

    df = add_prior_cumulative_stats(df)
    df = add_prior_efficiency_metrics(df)
    df = add_prior_derived_metrics(df)
    df = add_prior_weighted_average_metrics(df)
    df = add_daily_percentiles(
        df,
        RATING_METRICS,
    )

    df = add_skill_ratings(df)

    print(
        df[
            [
                "personId",
                "gameDate",
                "priorGames",
                "finishing",
                "shooting",
                "playmaking",
                "defense",
                "rebounding",
            ]
        ].head(30)
    )


if __name__ == "__main__":
    main()