from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo


PROCESSED_DATA_DIR = Path("data/processed")

PLAYER_GAME_PATHS = [
    PROCESSED_DATA_DIR / "player_games_2023_24.csv",
    PROCESSED_DATA_DIR / "player_games_2024_25.csv",
    PROCESSED_DATA_DIR / "player_games_2025_26.csv",
]

OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "player_profiles.csv"
)

CHECKPOINT_PATH = (
    PROCESSED_DATA_DIR
    / "player_profiles_checkpoint.csv"
)

def fetch_player_profile(person_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(
                player_id=person_id
            )

            return player_info.get_data_frames()[0]

        except Exception as e:
            print(
                f"Failed for {person_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )

            time.sleep(2)

    return None

def height_to_inches(height):
    if pd.isna(height):
        return None

    height = str(height).strip()

    if "-" not in height:
        return None

    feet, inches = height.split("-", 1)

    try:
        return int(feet) * 12 + int(inches)
    except ValueError:
        return None

def clean_player_profile(profile_df):
    row = profile_df.iloc[0]

    return {
        "personId": row["PERSON_ID"],
        "firstName": row["FIRST_NAME"],
        "familyName": row["LAST_NAME"],
        "position": row["POSITION"],
        "heightInches": height_to_inches(
            row["HEIGHT"]
        ),
        "weight": pd.to_numeric(
            row["WEIGHT"],
            errors="coerce",
        ),
        "birthDate": pd.to_datetime(
            row["BIRTHDATE"],
            errors="coerce",
        ),
    }

def fetch_all_player_profiles(player_ids):
    if CHECKPOINT_PATH.exists():
        checkpoint_df = pd.read_csv(
            CHECKPOINT_PATH
        )

        completed_ids = set(
            checkpoint_df["personId"].astype(int)
        )
    else:
        completed_ids = set()

    missing_ids = [
        int(person_id)
        for person_id in player_ids
        if int(person_id) not in completed_ids
    ]

    print(
        f"Total players: {len(player_ids)}"
    )
    print(
        f"Already completed: {len(completed_ids)}"
    )
    print(
        f"Need to fetch: {len(missing_ids)}"
    )

    for i, person_id in enumerate(
        missing_ids,
        start=1,
    ):
        print(
            f"{i}/{len(missing_ids)} - {person_id}"
        )

        profile_df = fetch_player_profile(
            person_id
        )

        if profile_df is None:
            print(f"Skipping {person_id}")
            continue

        cleaned_profile = clean_player_profile(
            profile_df
        )

        pd.DataFrame(
            [cleaned_profile]
        ).to_csv(
            CHECKPOINT_PATH,
            mode="a",
            header=not CHECKPOINT_PATH.exists(),
            index=False,
        )

        time.sleep(0.6)

    return pd.read_csv(
        CHECKPOINT_PATH
    )

if __name__ == "__main__":
    player_games_df = pd.concat(
        [
            pd.read_csv(path)
            for path in PLAYER_GAME_PATHS
        ],
        ignore_index=True,
    )

    player_ids = (
        player_games_df["personId"]
        .dropna()
        .astype(int)
        .unique()
    )

    profiles_df = fetch_all_player_profiles(
        player_ids
    )

    profiles_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(profiles_df.shape)
    print(f"Saved to {OUTPUT_PATH}")