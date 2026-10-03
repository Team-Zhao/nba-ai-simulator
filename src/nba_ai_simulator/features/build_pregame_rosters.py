from pathlib import Path
import pandas as pd
from nba_ai_simulator.features.build_rolling_team_features import (
    add_minutes_numeric,
    add_expected_minutes,
    add_expected_minutes_fallback,
)

PROCESSED_DATA_DIR = Path("data/processed")

HISTORY_SEASONS = [
    "2023_24",
    "2024_25",
    "2025_26",
]


def load_player_history():
    frames = []

    for season in HISTORY_SEASONS:
        path = (
            PROCESSED_DATA_DIR
            / f"player_games_{season}.csv"
        )

        df = pd.read_csv(
            path,
            dtype={"gameId": str},
            parse_dates=["gameDate"],
        )

        df["gameId"] = df["gameId"].str.zfill(10)
        df["season"] = season

        frames.append(df)

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history = history.sort_values(
        ["gameDate", "gameId", "personId"]
    ).reset_index(drop=True)

    return history


def build_pregame_roster(
    history,
    team,
    prediction_date,
):
    prediction_date = pd.Timestamp(prediction_date)

    # Only use games before the prediction date
    past = history[
        history["gameDate"] < prediction_date
    ].copy()

    # Find every player's latest historical appearance
    latest = (
        past.sort_values(
            ["gameDate", "gameId"]
        )
        .drop_duplicates(
            subset=["personId"],
            keep="last",
        )
    )

    # A player's latest team is our historical
    # estimate of their current team
    roster = latest[
        latest["teamTricode"] == team
    ].copy()

    roster = roster.rename(
        columns={
            "gameDate": "lastSeenDate",
            "gameId": "lastSeenGameId",
        }
    )

    roster["daysSinceLastSeen"] = (
        prediction_date - roster["lastSeenDate"]
    ).dt.days

    roster = roster.sort_values(
        "lastSeenDate",
        ascending=False,
    ).reset_index(drop=True)

    return roster


def add_recent_appearances(
    roster,
    history,
    team,
    prediction_date,
    n_games=10,
):
    prediction_date = pd.Timestamp(prediction_date)

    # Only games before prediction date
    team_history = history[
        (history["teamTricode"] == team)
        & (history["gameDate"] < prediction_date)
    ].copy()

    # Find this team's most recent N games
    recent_games = (
        team_history[
            ["gameId", "gameDate"]
        ]
        .drop_duplicates()
        .sort_values(["gameDate", "gameId"])
        .tail(n_games)
    )

    # Count how often each player appeared
    recent_player_games = team_history[
        team_history["gameId"].isin(
            recent_games["gameId"]
        )
    ]

    appearances = (
        recent_player_games
        .groupby("personId")["gameId"]
        .nunique()
    )

    roster = roster.copy()

    roster["appearancesLast10TeamGames"] = (
        roster["personId"]
        .map(appearances)
        .fillna(0)
        .astype(int)
    )

    return roster


def attach_expected_minutes(roster, history, prediction_date):
    prediction_date = pd.Timestamp(prediction_date)

    past = history[
        history["gameDate"] < prediction_date
    ].copy()

    # Reuse the original minutes converter
    past = add_minutes_numeric(past)

    # Add one temporary prediction row per roster candidate
    prediction_rows = pd.DataFrame({
        "personId": roster["personId"].to_numpy(),
        "gameDate": prediction_date,
        "minutesNumeric": 0.0,
        "isPredictionRow": True,
    })

    past["isPredictionRow"] = False

    combined = pd.concat(
        [past, prediction_rows],
        ignore_index=True,
    )

    # Reuse the original rolling 10-game calculation
    combined = add_expected_minutes(combined)
    combined = add_expected_minutes_fallback(combined)

    expected = combined.loc[
        combined["isPredictionRow"],
        ["personId", "expectedMinutes"],
    ]

    roster = roster.merge(
        expected,
        on="personId",
        how="left",
        validate="one_to_one",
    )

    assert roster["expectedMinutes"].notna().all()

    return roster

def add_roster_status(roster):
    roster = roster.copy()

    roster["rosterStatus"] = "uncertain"

    roster.loc[
        roster["appearancesLast10TeamGames"] >= 3,
        "rosterStatus"
    ] = "recent_rotation"

    roster.loc[
        (roster["appearancesLast10TeamGames"] == 0)
        & (roster["daysSinceLastSeen"] > 60),
        "rosterStatus"
    ] = "stale_candidate"

    return roster

def add_roster_status(roster):
    roster = roster.copy()

    roster["rosterStatus"] = "uncertain"

    # Recently part of the team's rotation
    roster.loc[
        roster["appearancesLast10TeamGames"] >= 3,
        "rosterStatus",
    ] = "recent_rotation"

    # Has not appeared recently and has been unseen for a long time
    roster.loc[
        (roster["appearancesLast10TeamGames"] == 0)
        & (roster["daysSinceLastSeen"] > 60),
        "rosterStatus",
    ] = "stale_candidate"

    return roster

def apply_availability_adjustment(roster):
    roster = roster.copy()

    roster["expectedMinutesAdjusted"] = (
        roster["expectedMinutes"]
    )

    roster.loc[
        roster["availabilityStatus"] == "OUT",
        "expectedMinutesAdjusted",
    ] = 0.0

    roster.loc[
        roster["availabilityStatus"] == "DOUBTFUL",
        "expectedMinutesAdjusted",
    ] *= 0.25

    roster.loc[
        roster["availabilityStatus"] == "QUESTIONABLE",
        "expectedMinutesAdjusted",
    ] *= 0.5

    return roster

def main():
    history = load_player_history()

    print("Shape:", history.shape)
    print("Unique games:", history["gameId"].nunique())
    print("Unique players:", history["personId"].nunique())

    print(
        "Date range:",
        history["gameDate"].min(),
        "to",
        history["gameDate"].max(),
    )

    duplicate_keys = history.duplicated(
        ["gameId", "personId"]
    ).sum()

    print("Duplicate player-game keys:", duplicate_keys)
    print("Missing dates:", history["gameDate"].isna().sum())

    print("\nRows by season:")
    print(history["season"].value_counts().sort_index())

    assert duplicate_keys == 0
    assert history["gameDate"].notna().all()

    roster = build_pregame_roster(
        history=history,
        team="LAL",
        prediction_date="2025-01-15",
    )


    roster = add_recent_appearances(
        roster=roster,
        history=history,
        team="LAL",
        prediction_date="2025-01-15",
        n_games=10,
    )

    roster = attach_expected_minutes(
        roster=roster,
        history=history,
        prediction_date="2025-01-15",
    )

    roster = add_roster_status(roster)
    roster["availabilityStatus"] = "UNKNOWN"
    roster["availabilitySource"] = None
    roster["availabilityUpdatedAt"] = pd.NaT

    roster = apply_availability_adjustment(roster)
    print("\nPregame roster status:")

    print(
        roster[
            [
                "personId",
                "rosterStatus",
                "expectedMinutes",
                "availabilityStatus",
                "expectedMinutesAdjusted",
            ]
        ]
        .sort_values(
            "expectedMinutes",
            ascending=False,
        )
        .to_string(index=False)
    )
    assert roster["rosterStatus"].isin(
        [
            "recent_rotation",
            "uncertain",
            "stale_candidate",
        ]
    ).all()

    assert roster["expectedMinutes"].notna().all()

    print("\nPregame roster with expected minutes:")

    print(
        roster[
            [
                "personId",
                "daysSinceLastSeen",
                "appearancesLast10TeamGames",
                "expectedMinutes",
            ]
        ]
        .sort_values(
            "expectedMinutes",
            ascending=False,
        )
        .to_string(index=False)
    )

    print("\nLAL roster with recent appearances:")

    print(
        roster[
            [
                "personId",
                "lastSeenDate",
                "daysSinceLastSeen",
                "appearancesLast10TeamGames",
            ]
        ]
        .sort_values(
            [
                "appearancesLast10TeamGames",
                "lastSeenDate",
            ],
            ascending=[False, False],
        )
        .to_string(index=False)
    )

    assert roster[
        "appearancesLast10TeamGames"
    ].between(0, 10).all()

    print("\nLAL historical roster candidates:")
    print("Prediction date: 2025-01-15")
    print("Roster candidates:", len(roster))

    print(
        roster[
            [
                "personId",
                "teamTricode",
                "lastSeenDate",
                "daysSinceLastSeen",
            ]
        ].to_string(index=False)
    )

    assert (
        roster["lastSeenDate"]
        < pd.Timestamp("2025-01-15")
    ).all()

    assert roster["personId"].is_unique
    assert roster["teamTricode"].eq("LAL").all()


if __name__ == "__main__":
    main()