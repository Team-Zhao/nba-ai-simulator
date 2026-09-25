from pathlib import Path

import pandas as pd


RATINGS_PATH = (
    Path("data/processed")
    / "player_ratings_rolling.csv"
)

PLAYER_GAMES_PATH = (
    Path("data/processed")
    / "player_games_two_seasons.csv"
)

TEAM_OUTPUT_PATH = (
    Path("data/processed")
    / "team_ratings_rolling.csv"
)

RECENT_GAMES = 10


def load_data():
    ratings = pd.read_csv(
        RATINGS_PATH,
        parse_dates=["gameDate"],
        dtype={"gameId": str},
    )

    games = pd.read_csv(
        PLAYER_GAMES_PATH,
        parse_dates=["gameDate"],
        dtype={"gameId": str},
    )

    ratings["gameId"] = ratings["gameId"].str.zfill(10)
    games["gameId"] = games["gameId"].str.zfill(10)

    return ratings, games

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

def add_expected_minutes(df):
    df = df.copy()

    df = df.sort_values(
        ["personId", "gameDate"]
    )

    df["expectedMinutes"] = (
        df.groupby("personId")["minutesNumeric"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    RECENT_GAMES,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    return df

RATING_COLS = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
]


def add_cold_start_ratings(df):
    df = df.copy()

    skill_cols = [
        "finishing",
        "shooting",
        "playmaking",
        "defense",
        "rebounding",
    ]

    df["isColdStart"] = (
        df[skill_cols]
        .isna()
        .all(axis=1)
    )

    df[skill_cols] = (
        df[skill_cols]
        .fillna(50.0)
    )

    df["physical"] = (
        df["physical"]
        .fillna(50.0)
    )

    return df

DEFAULT_EXPECTED_MINUTES = 20.0

def add_expected_minutes_fallback(df):
    df = df.copy()

    df["expectedMinutes"] = (
        df["expectedMinutes"]
        .fillna(DEFAULT_EXPECTED_MINUTES)
    )

    return df

def build_team_game_ratings(df):
    df = df.copy()

    for rating in RATING_COLS:
        df[f"{rating}Weighted"] = (
            df[rating] * df["expectedMinutes"]
        )

    agg_dict = {
        "expectedMinutesTotal": (
            "expectedMinutes",
            "sum",
        ),
    }

    for rating in RATING_COLS:
        agg_dict[f"{rating}WeightedTotal"] = (
            f"{rating}Weighted",
            "sum",
        )

    team_df = (
        df.groupby(
            ["gameId", "gameDate", "teamTricode"],
            as_index=False,
        )
        .agg(**agg_dict)
    )

    for rating in RATING_COLS:
        team_df[rating] = (
            team_df[f"{rating}WeightedTotal"]
            / team_df["expectedMinutesTotal"]
        )

    return team_df[
        [
            "gameId",
            "gameDate",
            "teamTricode",
            "expectedMinutesTotal",
            *RATING_COLS,
        ]
    ]


def main():
    ratings, games = load_data()

    games = add_minutes_numeric(games)
    games = add_expected_minutes(games)

    minutes_df = games[
        [
            "gameId",
            "personId",
            "expectedMinutes",
        ]
    ].copy()

    df = ratings.merge(
        minutes_df,
        on=["gameId", "personId"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    assert (
        df["_merge"] == "both"
    ).all(), "Some player ratings failed to match player games"

    df = df.drop(columns="_merge")

    print("Player-game rows:", len(df))
    print(
        "Missing expected minutes before fallback:",
        df["expectedMinutes"].isna().sum(),
    )

    df = add_cold_start_ratings(df)
    df = add_expected_minutes_fallback(df)

    assert df["expectedMinutes"].notna().all()
    assert df[RATING_COLS].notna().all().all()

    print("Cold-start rows:", df["isColdStart"].sum())

    team_df = build_team_game_ratings(df)

    assert (
        team_df.groupby("gameId")["teamTricode"]
        .nunique()
        .eq(2)
        .all()
    ), "Some games do not have exactly two teams"

    assert (
        team_df["expectedMinutesTotal"] > 0
    ).all(), "Found non-positive team minutes"

    assert (
        team_df[RATING_COLS]
        .notna()
        .all()
        .all()
    ), "Found missing team ratings"

    team_df.to_csv(
        TEAM_OUTPUT_PATH,
        index=False,
    )

    print(f"Saved to {TEAM_OUTPUT_PATH}")
    print("Team features shape:", team_df.shape)
    print("Unique games:", team_df["gameId"].nunique())
    print(team_df.head(10))


if __name__ == "__main__":
    main()