from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")


def build_two_season_dataset():
    df_2024 = pd.read_csv(
        PROCESSED_DATA_DIR / "player_games_2024_25.csv",
        dtype={"gameId": str},
    )

    df_2025 = pd.read_csv(
        PROCESSED_DATA_DIR / "player_games_2025_26.csv",
        dtype={"gameId": str},
    )

    player_games_df = pd.concat(
        [df_2024, df_2025],
        ignore_index=True,
    )

    player_games_df["gameDate"] = pd.to_datetime(
        player_games_df["gameDate"]
    )

    player_games_df = (
        player_games_df
        .sort_values("gameDate")
        .reset_index(drop=True)
    )

    return player_games_df


if __name__ == "__main__":
    player_games_df = build_two_season_dataset()

    output_path = (
        PROCESSED_DATA_DIR
        / "player_games_two_seasons.csv"
    )

    player_games_df.to_csv(
        output_path,
        index=False,
    )

    print(player_games_df.shape)
    print(player_games_df["season"].value_counts())
    print("Games:", player_games_df["gameId"].nunique())
    print("From:", player_games_df["gameDate"].min())
    print("To:", player_games_df["gameDate"].max())
    print(f"Saved to {output_path}")