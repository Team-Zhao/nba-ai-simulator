from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


PROCESSED_DATA_DIR = Path("data/processed")

TEAM_RATINGS_PATH = (
    PROCESSED_DATA_DIR
    / "team_ratings_rolling.csv"
)

SEASONS = [
    "2024-25",
    "2025-26",
]

RATING_COLS = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
]

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "game_features_rolling.csv"
)

HOME_AWAY_OVERRIDES = {
    # 2024-25
    "0022400147": {"homeTeam": "MIA", "awayTeam": "WAS"},
    "0022400621": {"homeTeam": "IND", "awayTeam": "SAS"},
    "0022400633": {"homeTeam": "SAS", "awayTeam": "IND"},
    "0022401229": {"homeTeam": "MIL", "awayTeam": "ATL"},
    "0022401230": {"homeTeam": "OKC", "awayTeam": "HOU"},

    # 2025-26
    "0022500147": {"homeTeam": "DET", "awayTeam": "DAL"},
    "0022500578": {"homeTeam": "ORL", "awayTeam": "MEM"},
    "0022500602": {"homeTeam": "MEM", "awayTeam": "ORL"},
    "0022501229": {"homeTeam": "ORL", "awayTeam": "NYK"},
    "0022501230": {"homeTeam": "OKC", "awayTeam": "SAS"},
}


def load_team_ratings():
    team_df = pd.read_csv(
        TEAM_RATINGS_PATH,
        dtype={"gameId": str},
        parse_dates=["gameDate"],
    )

    team_df["gameId"] = (
        team_df["gameId"].str.zfill(10)
    )

    return team_df


def fetch_game_logs():
    frames = []

    for season in SEASONS:
        print(f"Fetching game logs: {season}")

        game_log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
        )

        games = game_log.get_data_frames()[0]

        games["season"] = season

        frames.append(games)

    games_df = pd.concat(
        frames,
        ignore_index=True,
    )

    games_df["gameId"] = (
        games_df["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    return games_df


def check_home_away(games_df, team_df):
    locations = games_df[
        [
            "gameId",
            "TEAM_ABBREVIATION",
            "MATCHUP",
        ]
    ].copy()

    # Only inspect games in our rolling dataset
    locations = locations[
        locations["gameId"].isin(
            team_df["gameId"]
        )
    ].copy()

    locations["isHome"] = (
        locations["MATCHUP"]
        .str.contains(
            "vs.",
            regex=False,
            na=False,
        )
    )

    game_check = (
        locations.groupby("gameId")["isHome"]
        .agg(
            rows="size",
            homeCount="sum",
        )
    )

    bad_games = game_check[
        (game_check["rows"] != 2)
        | (game_check["homeCount"] != 1)
    ]

    print("Games checked:", len(game_check))
    print("Bad home/away games:", len(bad_games))

    if not bad_games.empty:
        print("\nProblematic matchups:")

        print(
            locations[
                locations["gameId"].isin(
                    bad_games.index
                )
            ][
                [
                    "gameId",
                    "TEAM_ABBREVIATION",
                    "MATCHUP",
                    "isHome",
                ]
            ].sort_values("gameId").to_string(index=False)
        )

    return bad_games


def add_home_away(team_df, games_df):
    locations = games_df[
        [
            "gameId",
            "TEAM_ABBREVIATION",
            "MATCHUP",
        ]
    ].copy()

    locations = locations.rename(
        columns={
            "TEAM_ABBREVIATION": "teamTricode",
        }
    )

    locations["isHome"] = (
        locations["MATCHUP"]
        .str.contains("vs.", regex=False, na=False)
    )

    # Apply verified overrides
    for game_id, teams in HOME_AWAY_OVERRIDES.items():
        locations.loc[
            (locations["gameId"] == game_id)
            & (locations["teamTricode"] == teams["homeTeam"]),
            "isHome",
        ] = True

        locations.loc[
            (locations["gameId"] == game_id)
            & (locations["teamTricode"] == teams["awayTeam"]),
            "isHome",
        ] = False

    locations = locations[
        ["gameId", "teamTricode", "isHome"]
    ]

    team_df = team_df.merge(
        locations,
        on=["gameId", "teamTricode"],
        how="left",
        validate="one_to_one",
    )

    # Validate every game has exactly one home team
    assert team_df["isHome"].notna().all(), (
        "Some teams are missing home/away information"
    )

    game_check = (
        team_df.groupby("gameId")["isHome"]
        .agg(
            rows="size",
            homeCount="sum",
        )
    )

    assert (
        (game_check["rows"] == 2)
        & (game_check["homeCount"] == 1)
    ).all(), "Invalid home/away assignments"

    return team_df


def build_game_matchup_features(team_df):
    home_df = team_df[
        team_df["isHome"]
    ].copy()

    away_df = team_df[
        ~team_df["isHome"]
    ].copy()

    home_rename = {
        "teamTricode": "homeTeam",
        "expectedMinutesTotal": "homeExpectedMinutesTotal",
    }

    away_rename = {
        "teamTricode": "awayTeam",
        "expectedMinutesTotal": "awayExpectedMinutesTotal",
    }

    for rating in RATING_COLS:
        home_rename[rating] = f"home{rating.capitalize()}"
        away_rename[rating] = f"away{rating.capitalize()}"

    home_df = home_df.rename(columns=home_rename)
    away_df = away_df.rename(columns=away_rename)

    # gameDate only needs to appear once
    away_df = away_df.drop(columns=["gameDate"])

    game_df = home_df.merge(
        away_df,
        on="gameId",
        how="inner",
        validate="one_to_one",
    )

    return game_df

def add_rating_differences(game_df):
    game_df = game_df.copy()

    for rating in RATING_COLS:
        name = rating.capitalize()

        game_df[f"{rating}Diff"] = (
            game_df[f"home{name}"]
            - game_df[f"away{name}"]
        )

    return game_df


def add_game_outcomes(game_df, games_df):
    game_df = game_df.copy()

    outcomes = games_df[
        [
            "gameId",
            "TEAM_ABBREVIATION",
            "PTS",
        ]
    ].copy()

    home_scores = outcomes.rename(
        columns={
            "TEAM_ABBREVIATION": "homeTeam",
            "PTS": "homeScore",
        }
    )[
        ["gameId", "homeTeam", "homeScore"]
    ]

    away_scores = outcomes.rename(
        columns={
            "TEAM_ABBREVIATION": "awayTeam",
            "PTS": "awayScore",
        }
    )[
        ["gameId", "awayTeam", "awayScore"]
    ]

    game_df = game_df.merge(
        home_scores,
        on=["gameId", "homeTeam"],
        how="left",
        validate="one_to_one",
    )

    game_df = game_df.merge(
        away_scores,
        on=["gameId", "awayTeam"],
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

    return game_df



def main():
    team_df = load_team_ratings()

    games_df = fetch_game_logs()

    print("Team rows:", len(team_df))
    print("Unique team games:", team_df["gameId"].nunique())

    print("Game log rows:", len(games_df))
    print("Unique game log games:", games_df["gameId"].nunique())

    missing_ids = (
        set(team_df["gameId"])
        - set(games_df["gameId"])
    )

    print("Missing game IDs:", len(missing_ids))

    if missing_ids:
        print(sorted(missing_ids)[:20])
        raise ValueError("Game logs are missing required games")

    check_home_away(
        games_df,
        team_df,
    )

    team_df = add_home_away(
        team_df,
        games_df,
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

    diff_cols = [
        f"{rating}Diff"
        for rating in RATING_COLS
    ]

    assert len(game_df) == 2460
    assert game_df["gameId"].is_unique

    assert game_df[
        ["homeScore", "awayScore", "homePointDiff"]
    ].notna().all().all()

    assert game_df[diff_cols].notna().all().all()

    assert game_df["gameDate"].notna().all()

    assert game_df["homeWin"].isin([0, 1]).all()

    assert (
        game_df["homePointDiff"]
        == game_df["homeScore"] - game_df["awayScore"]
    ).all()

    print("\nGame features shape:", game_df.shape)
    print("Unique games:", game_df["gameId"].nunique())

    print(
        game_df[
            [
                "gameId",
                "gameDate",
                "homeTeam",
                "awayTeam",
                *diff_cols,
            ]
        ].head(10)
    )

    print("\nAfter home/away correction:")

    print(
        team_df.groupby("gameId")["isHome"]
        .sum()
        .value_counts()
    )

    print(
        team_df[
            team_df["gameId"].isin(HOME_AWAY_OVERRIDES)
        ][
            ["gameId", "teamTricode", "isHome"]
        ]
        .sort_values(["gameId", "isHome"])
        .to_string(index=False)
    )


    game_df = game_df.sort_values(
        ["gameDate", "gameId"]
    ).reset_index(drop=True)

    game_df["season"] = (
        game_df["gameId"]
        .str[3:5]
        .astype(int)
        .map({
            24: "2024-25",
            25: "2025-26",
        })
    )

    assert game_df["season"].notna().all()

    game_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved to {OUTPUT_PATH}")
    print("Shape:", game_df.shape)
    print("Unique games:", game_df["gameId"].nunique())

    print(
        game_df[
            [
                "gameId",
                "gameDate",
                "homeTeam",
                "awayTeam",
                "homeScore",
                "awayScore",
                "homePointDiff",
                "homeWin",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()