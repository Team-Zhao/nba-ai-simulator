from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")


def build_train_window():
    df_2023 = pd.read_csv(
        PROCESSED_DATA_DIR / "player_games_2023_24.csv",
        dtype={"gameId": str},
    )

    df_2024 = pd.read_csv(
        PROCESSED_DATA_DIR / "player_games_2024_25.csv",
        dtype={"gameId": str},
    )

    df_2025 = pd.read_csv(
        PROCESSED_DATA_DIR / "player_games_2025_26.csv",
        dtype={"gameId": str},
    )

    train_window_df = pd.concat(
        [df_2023, df_2024],
        ignore_index=True,
    )  

    train_window_df["gameDate"] = pd.to_datetime(
        train_window_df["gameDate"]
    )

    # player_games_df = pd.concat(
    #     [df_2024, df_2025],
    #     ignore_index=True,
    # )

    # player_games_df["gameDate"] = pd.to_datetime(
    #     player_games_df["gameDate"]
    # )

    # player_games_df = (
    #     player_games_df
    #     .sort_values("gameDate")
    #     .reset_index(drop=True)
    # )

    # return player_games_df
    train_window_df = (
        train_window_df
        .sort_values("gameDate")
        .reset_index(drop=True)
    )

    return train_window_df


if __name__ == "__main__":
    train_window_df = build_train_window()

    output_path = (
        PROCESSED_DATA_DIR
        / "player_games_train_window.csv"
    )

    train_window_df.to_csv(
        output_path,
        index=False,
    )

    print(train_window_df.shape)
    print(train_window_df["season"].value_counts())
    print("Games:", train_window_df["gameId"].nunique())
    print("From:", train_window_df["gameDate"].min())
    print("To:", train_window_df["gameDate"].max())
    print(f"Saved to {output_path}")