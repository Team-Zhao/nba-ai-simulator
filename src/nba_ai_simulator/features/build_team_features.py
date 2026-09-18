import numpy as np
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog
from nba_api.live.nba.endpoints import boxscore

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

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "game_features_2025_26.csv"
)

RATING_COLS = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
]

HOME_AWAY_OVERRIDES = {
    "0022500147": {
        "homeTeam": "DET",
        "awayTeam": "DAL",
    },
    "0022500578": {
        "homeTeam": "ORL",
        "awayTeam": "MEM",
    },
    "0022500602": {
        "homeTeam": "MEM",
        "awayTeam": "ORL",
    },
    "0022501229": {
        "homeTeam": "ORL",
        "awayTeam": "NYK",
    },
    "0022501230": {
        "homeTeam": "OKC",
        "awayTeam": "SAS",
    },
}

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

def add_home_away(
    team_df,
    games_df,
    schedule_df=None,
):
    locations = games_df[
        [
            "GAME_ID",
            "TEAM_ABBREVIATION",
            "MATCHUP",
        ]
    ].copy()

    locations["gameId"] = (
        locations["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    locations["teamTricode"] = (
        locations["TEAM_ABBREVIATION"]
    )

    locations["isHome"] = (
        locations["MATCHUP"]
        .str.contains("vs.")
    )

    bad_games = (
        locations
        .groupby("gameId")["isHome"]
        .agg(
            rows="size",
            homeCount="sum",
        )
    )

    bad_games = bad_games[
        (bad_games["rows"] != 2)
        | (bad_games["homeCount"] != 1)
    ]

    bad_game_ids = set(
        bad_games.index
    )

    if schedule_df is not None and bad_game_ids:
        schedule_subset = schedule_df[
            schedule_df["gameId"].isin(
                bad_game_ids
            )
        ]

        for _, row in schedule_subset.iterrows():
            locations.loc[
                (locations["gameId"] == row["gameId"])
                & (
                    locations["teamTricode"]
                    == row["homeTeam"]
                ),
                "isHome",
            ] = True

            locations.loc[
                (locations["gameId"] == row["gameId"])
                & (
                    locations["teamTricode"]
                    == row["awayTeam"]
                ),
                "isHome",
            ] = False

    locations = locations[
        [
            "gameId",
            "teamTricode",
            "isHome",
        ]
    ]

    return team_df.merge(
        locations,
        on=[
            "gameId",
            "teamTricode",
        ],
        how="left",
        validate="one_to_one",
    )

def build_game_matchup_features(team_df):
    home_df = (
        team_df[
            team_df["isHome"]
        ]
        .copy()
    )

    away_df = (
        team_df[
            ~team_df["isHome"]
        ]
        .copy()
    )

    home_df = home_df.rename(
        columns={
            "teamTricode": "homeTeam",
            "teamFinishing": "homeFinishing",
            "teamShooting": "homeShooting",
            "teamPlaymaking": "homePlaymaking",
            "teamDefense": "homeDefense",
            "teamRebounding": "homeRebounding",
            "teamPhysical": "homePhysical",
        }
    )

    away_df = away_df.rename(
        columns={
            "teamTricode": "awayTeam",
            "teamFinishing": "awayFinishing",
            "teamShooting": "awayShooting",
            "teamPlaymaking": "awayPlaymaking",
            "teamDefense": "awayDefense",
            "teamRebounding": "awayRebounding",
            "teamPhysical": "awayPhysical",
        }
    )

    game_df = home_df.merge(
        away_df,
        on="gameId",
        how="inner",
        validate="one_to_one",
    )

    return game_df

def add_rating_differences(game_df):
    game_df = game_df.copy()

    for rating in [
        "Finishing",
        "Shooting",
        "Playmaking",
        "Defense",
        "Rebounding",
        "Physical",
    ]:
        game_df[
            f"{rating.lower()}Diff"
        ] = (
            game_df[f"home{rating}"]
            - game_df[f"away{rating}"]
        )

    return game_df

def add_game_outcomes(game_df, games_df):
    outcomes = games_df[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ABBREVIATION",
            "PTS",
        ]
    ].copy()

    game_dates = (
        outcomes[
            [
                "GAME_ID",
                "GAME_DATE",
            ]
        ]
        .drop_duplicates("GAME_ID")
        .rename(
            columns={
                "GAME_ID": "gameId",
                "GAME_DATE": "gameDate",
            }
        )
    )

    game_dates["gameId"] = (
        game_dates["gameId"]
        .astype(str)
        .str.zfill(10)
    )

    outcomes["gameId"] = (
        outcomes["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    home_scores = outcomes.rename(
        columns={
            "TEAM_ABBREVIATION": "homeTeam",
            "PTS": "homeScore",
        }
    )[
        [
            "gameId",
            "homeTeam",
            "homeScore",
        ]
    ]

    away_scores = outcomes.rename(
        columns={
            "TEAM_ABBREVIATION": "awayTeam",
            "PTS": "awayScore",
        }
    )[
        [
            "gameId",
            "awayTeam",
            "awayScore",
        ]
    ]

    game_df = game_df.merge(
        home_scores,
        on=[
            "gameId",
            "homeTeam",
        ],
        how="left",
        validate="one_to_one",
    )

    game_df = game_df.merge(
        away_scores,
        on=[
            "gameId",
            "awayTeam",
        ],
        how="left",
        validate="one_to_one",
    )

    game_df["homePointDiff"] = (
        game_df["homeScore"]
        - game_df["awayScore"]
    )

    game_df["homeWin"] = (
        game_df["homePointDiff"] > 0
    ).astype(int)

    game_df = game_df.merge(
        game_dates,
        on="gameId",
        how="left",
        validate="one_to_one",
    )

    game_df["gameDate"] = pd.to_datetime(
        game_df["gameDate"]
    )

    return game_df

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

# def fetch_schedule_home_away():
#     import requests

#     url = (
#         "https://cdn.nba.com/static/json/"
#         "staticData/scheduleLeagueV2.json"
#     )

#     schedule = requests.get(url).json()

#     rows = []

#     for date_block in schedule[
#         "leagueSchedule"
#     ]["gameDates"]:

#         for game in date_block["games"]:
#             rows.append({
#                 "gameId": str(
#                     game["gameId"]
#                 ).zfill(10),

#                 "homeTeam": (
#                     game["homeTeam"]["teamTricode"]
#                 ),

#                 "awayTeam": (
#                     game["awayTeam"]["teamTricode"]
#                 ),
#             })

#     return pd.DataFrame(rows)

def fetch_home_away_fallback(
    bad_game_ids,
    games_df,
):
    rows = []

    for game_id in bad_game_ids:
        if game_id not in HOME_AWAY_OVERRIDES:
            raise ValueError(
                f"No home/away override for {game_id}"
            )

        rows.append({
            "gameId": game_id,
            **HOME_AWAY_OVERRIDES[game_id],
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = prepare_team_feature_data()

    team_df = build_team_game_ratings(
        df
    )

    game_log = leaguegamelog.LeagueGameLog(
        season="2025-26",
        season_type_all_star="Regular Season",
    )

    games_df = game_log.get_data_frames()[0]

    locations_check = games_df.copy()

    locations_check["gameId"] = (
        locations_check["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    locations_check["isHome"] = (
        locations_check["MATCHUP"]
        .str.contains("vs.")
    )

    bad_games = (
        locations_check
        .groupby("gameId")["isHome"]
        .agg(
            rows="size",
            homeCount="sum",
        )
    )

    bad_games = bad_games[
        (bad_games["rows"] != 2)
        | (bad_games["homeCount"] != 1)
    ]

    schedule_df = fetch_home_away_fallback(
        bad_games.index.tolist(),
        games_df,
    )

    team_df = add_home_away(
        team_df,
        games_df,
        schedule_df,
    )

    print(bad_games)

    print(
        team_df[
            team_df["gameId"].isin(
                bad_games.index
            )
        ][
            [
                "gameId",
                "teamTricode",
                "isHome",
            ]
        ]
        .sort_values("gameId")
    )

    bad_game_ids = set(
        bad_games.index
    )

    check_games = games_df.copy()

    check_games["gameId"] = (
        check_games["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    print(
        check_games[
            check_games["gameId"].isin(
                bad_game_ids
            )
        ][
            [
                "gameId",
                "TEAM_ABBREVIATION",
                "MATCHUP",
            ]
        ]
        .sort_values("gameId")
        .to_string(index=False)
    )

    game_df = build_game_matchup_features(
        team_df
    )

    game_df = add_rating_differences(
        game_df
    )

    game_df = add_game_outcomes(
        game_df,
        games_df,
    )

    print(game_df.shape)

    bad_game_ids = set(
        bad_games.index
    )

    check_games = games_df.copy()

    check_games["gameId"] = (
        check_games["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    print(
        check_games[
            check_games["gameId"].isin(
                bad_game_ids
            )
        ][
            [
                "gameId",
                "TEAM_ABBREVIATION",
                "MATCHUP",
            ]
        ]
        .sort_values("gameId")
        .to_string(index=False)
    )

    print(
        game_df[
            [
                "gameId",
                "homeTeam",
                "awayTeam",
                "homeScore",
                "awayScore",
                "homePointDiff",
                "homeWin",
                "finishingDiff",
                "shootingDiff",
                "playmakingDiff",
                "defenseDiff",
                "reboundingDiff",
                "physicalDiff",
            ]
        ].head(20)
    )

    print(
        game_df[
            [
                "homeScore",
                "awayScore",
                "homePointDiff",
            ]
        ]
        .isna()
        .sum()
    )

    game_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved to {OUTPUT_PATH}")
    print("Games:", game_df["gameId"].nunique())

    print(
        game_df[
            [
                "finishingDiff",
                "shootingDiff",
                "playmakingDiff",
                "defenseDiff",
                "reboundingDiff",
                "physicalDiff",
                "homePointDiff",
            ]
        ].describe()
    )

    print(
        game_df[
            "homePointDiff"
        ].describe()
    )

    print(
        "Home win rate:",
        game_df["homeWin"].mean()
    )

    print(
        game_df[
            [
                "gameId",
                "gameDate",
                "homeTeam",
                "awayTeam",
                "homePointDiff",
            ]
        ].head()
    )

    print(
        "Missing gameDate:",
        game_df["gameDate"].isna().sum()
    )