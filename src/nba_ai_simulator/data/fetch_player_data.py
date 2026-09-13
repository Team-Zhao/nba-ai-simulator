from pathlib import Path
import time
import pandas as pd

from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEASONS = [
    "2024-25",
    "2025-26",
]

def get_checkpoint_paths(season):
    season_slug = season.replace("-", "_")

    traditional_path = (
        RAW_DATA_DIR
        / f"traditional_player_games_{season_slug}.csv"
    )

    advanced_path = (
        RAW_DATA_DIR
        / f"advanced_player_games_{season_slug}.csv"
    )

    completed_path = (
        RAW_DATA_DIR
        / f"completed_game_ids_{season_slug}.txt"
    )

    return traditional_path, advanced_path, completed_path

def load_completed_game_ids(completed_games_file):
    if not completed_games_file.exists():
        return set()

    with open(completed_games_file, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }

def save_completed_game_id(game_id, completed_games_file,):
    with open(completed_games_file, "a") as f:
        f.write(f"{game_id}\n")

def append_to_csv(df, path):
    file_exists = path.exists()

    df.to_csv(
        path,
        mode="a",
        header=not file_exists,
        index=False,
    )

def fetch_game_ids(season):
    game_log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    )

    games_df = game_log.get_data_frames()[0]

    game_ids = (
        games_df["GAME_ID"]
        .drop_duplicates()
        .tolist()
    )

    return games_df, game_ids

def fetch_traditional_box_score(game_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id
            )
            return box.get_data_frames()[0]

        except Exception as e:
            print(
                f"Traditional failed for {game_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )

            time.sleep(2)

    return None

def fetch_advanced_box_score(game_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            box = boxscoreadvancedv3.BoxScoreAdvancedV3(
                game_id=game_id
            )
            return box.get_data_frames()[0]

        except Exception as e:
            print(
                f"Advanced failed for {game_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )

            time.sleep(2)

    return None

def fetch_season_player_games(season, max_games=None):
    games_df, game_ids = fetch_game_ids(season)

    (traditional_checkpoint, advanced_checkpoint, completed_games_file,) = get_checkpoint_paths(season)

    completed_game_ids = load_completed_game_ids(
        completed_games_file
    )

    if max_games is not None:
        game_ids = game_ids[:max_games]

    for i, game_id in enumerate(game_ids, start=1):
        if game_id in completed_game_ids:
            print(f"Skipping completed game: {game_id}")
            continue
        print(f"[{season}] {i}/{len(game_ids)} - {game_id}")

        traditional_df = fetch_traditional_box_score(game_id)
        advanced_df = fetch_advanced_box_score(game_id)

        if traditional_df is None or advanced_df is None:
            print(f"Skipping {game_id}")
            continue

        append_to_csv(
            traditional_df,
            traditional_checkpoint,
        )
        append_to_csv(
            advanced_df,
            advanced_checkpoint,
        )

        save_completed_game_id(game_id,completed_games_file,)
        completed_game_ids.add(game_id)

        time.sleep(0.6)

    traditional_games_df = pd.read_csv(
        traditional_checkpoint,
        dtype={"gameId": str},
    )

    advanced_games_df = pd.read_csv(
        advanced_checkpoint,
        dtype={"gameId": str},
    )

    player_games_df = build_player_games(
        traditional_games_df,
        advanced_games_df,
        games_df,
        season,
    )

    return player_games_df

def clean_advanced_player_games(advanced_df):
    advanced_df = advanced_df.copy()

    advanced_df["has_minutes"] = (
        advanced_df["minutes"].notna()
    )

    advanced_df = (
        advanced_df
        .sort_values("has_minutes")
        .drop_duplicates(
            subset=["gameId", "personId"],
            keep="last",
        )
        .drop(columns="has_minutes")
        .reset_index(drop=True)
    )

    return advanced_df

def build_player_games(
    traditional_df,
    advanced_df,
    games_df,
    season,
):
    traditional_df = traditional_df.copy()
    advanced_df = advanced_df.copy()
    games_df = games_df.copy()

    traditional_df["gameId"] = (
        traditional_df["gameId"]
        .astype(str)
        .str.zfill(10)
    )

    advanced_df["gameId"] = (
        advanced_df["gameId"]
        .astype(str)
        .str.zfill(10)
    )

    games_df["GAME_ID"] = (
        games_df["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    advanced_df = clean_advanced_player_games(
        advanced_df
    )

    advanced_features = [
        "gameId",
        "personId",
        "effectiveFieldGoalPercentage",
        "trueShootingPercentage",
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
        "possessions",
        "PIE",
    ]

    player_games_df = traditional_df.merge(
        advanced_df[advanced_features],
        on=["gameId", "personId"],
        how="left",
        validate="one_to_one",
    )

    game_dates = (
        games_df[["GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .rename(
            columns={
                "GAME_ID": "gameId",
                "GAME_DATE": "gameDate",
            }
        )
    )

    player_games_df = player_games_df.merge(
        game_dates,
        on="gameId",
        how="left",
        validate="many_to_one",
    )

    player_games_df["gameDate"] = pd.to_datetime(
        player_games_df["gameDate"]
    )

    player_games_df["season"] = season

    return player_games_df

if __name__ == "__main__":
    player_games_df = fetch_season_player_games(
        "2025-26"
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "player_games_2025_26.csv"
    )

    player_games_df.to_csv(
        output_path,
        index=False,
    )

    print(player_games_df.shape)
    print(f"Saved to {output_path}")