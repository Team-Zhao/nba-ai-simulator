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

TRADITIONAL_CHECKPOINT = RAW_DATA_DIR / "traditional_player_games.csv"
ADVANCED_CHECKPOINT = RAW_DATA_DIR / "advanced_player_games.csv"
COMPLETED_GAMES_FILE = RAW_DATA_DIR / "completed_game_ids.txt"

SEASONS = [
    "2024-25",
    "2025-26",
]

def load_completed_game_ids():
    if not COMPLETED_GAMES_FILE.exists():
        return set()

    with open(COMPLETED_GAMES_FILE, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }

def save_completed_game_id(game_id):
    with open(COMPLETED_GAMES_FILE, "a") as f:
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
    completed_game_ids = load_completed_game_ids()

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
            TRADITIONAL_CHECKPOINT,
        )
        append_to_csv(
            advanced_df,
            ADVANCED_CHECKPOINT,
        )
        save_completed_game_id(game_id)
        completed_game_ids.add(game_id)

        time.sleep(0.6)

    traditional_games_df = pd.read_csv(
        TRADITIONAL_CHECKPOINT
    )

    advanced_games_df = pd.read_csv(
        ADVANCED_CHECKPOINT
    )

    return games_df, traditional_games_df, advanced_games_df

if __name__ == "__main__":
    games_df, traditional_df, advanced_df = fetch_season_player_games(
        "2024-25",
    )

    print("Traditional:", traditional_df.shape)
    print("Advanced:", advanced_df.shape)