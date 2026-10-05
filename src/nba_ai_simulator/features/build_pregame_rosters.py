from pathlib import Path
import pandas as pd
from nba_ai_simulator.features.build_rolling_team_features import (
    add_minutes_numeric,
    add_expected_minutes,
    add_expected_minutes_fallback,
    add_cold_start_ratings,
    build_team_game_ratings,
    RATING_COLS,
)
from datetime import datetime
from nba_ai_simulator.retrieval.injury_report_retriever import (
    retrieve_official_injury_updates,
    filter_updates_for_game,
    apply_availability_updates,
)
from nba_ai_simulator.features.build_rolling_game_features import (
    build_game_matchup_features,
    add_rating_differences,
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
    # expectedMinutesAdjusted is used as a relative weighting signal.
    # Its team total is not interpreted as literal playable minutes.
    # Removed injury minutes are implicitly redistributed proportionally
    # through weighted-average normalization.

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

def attach_pregame_player_ratings(
    roster,
    ratings,
    team,
    prediction_date,
):
    roster = roster.copy()
    ratings = ratings.copy()

    prediction_date = pd.Timestamp(
        prediction_date
    )

    # --------------------------------------------------
    # 1. Ratings from the actual target game
    # --------------------------------------------------

    same_day = ratings[
        (ratings["gameDate"] == prediction_date)
        & (ratings["teamTricode"] == team)
    ].copy()

    same_day = same_day[
        [
            "personId",
            *RATING_COLS,
        ]
    ]

    # There should only be one rating row per player
    same_day = same_day.drop_duplicates(
        subset=["personId"],
        keep="last",
    )

    roster = roster.merge(
        same_day,
        on="personId",
        how="left",
        validate="one_to_one",
    )

    roster["ratingSource"] = None

    same_day_mask = (
        roster[RATING_COLS]
        .notna()
        .any(axis=1)
    )

    roster.loc[
        same_day_mask,
        "ratingSource",
    ] = "target_game"

    # --------------------------------------------------
    # 2. Fallback for players absent from target game
    # --------------------------------------------------

    missing_mask = (
        roster[RATING_COLS]
        .isna()
        .all(axis=1)
    )

    if missing_mask.any():
        prior = ratings[
            ratings["gameDate"]
            < prediction_date
        ].copy()

        prior = (
            prior
            .sort_values(
                [
                    "personId",
                    "gameDate",
                ]
            )
            .drop_duplicates(
                subset=["personId"],
                keep="last",
            )
        )

        fallback = (
            roster.loc[
                missing_mask,
                ["personId"],
            ]
            .merge(
                prior[
                    [
                        "personId",
                        *RATING_COLS,
                    ]
                ],
                on="personId",
                how="left",
                validate="one_to_one",
            )
        )

        for rating in RATING_COLS:
            roster.loc[
                missing_mask,
                rating,
            ] = fallback[
                rating
            ].to_numpy()

        fallback_found = (
            fallback[RATING_COLS]
            .notna()
            .any(axis=1)
            .to_numpy()
        )

        missing_indices = (
            roster.index[missing_mask]
        )

        roster.loc[
            missing_indices[
                fallback_found
            ],
            "ratingSource",
        ] = "latest_prior"

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
        roster["expectedMinutesRosterAdjusted"]
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

def apply_roster_status_adjustment(roster):
    roster = roster.copy()

    roster["expectedMinutesRosterAdjusted"] = (
        roster["expectedMinutes"]
    )

    stale_mask = (
        roster["rosterStatus"]
        == "stale_candidate"
    )

    roster.loc[
        stale_mask,
        "expectedMinutesRosterAdjusted",
    ] = 0.0

    return roster

def main():
    history = load_player_history()

    ratings = pd.read_csv(
        "data/processed/player_ratings_rolling.csv",
        parse_dates=["gameDate"],
        dtype={"gameId": str},
    )

    ratings["gameId"] = (
        ratings["gameId"]
        .str.zfill(10)
    )

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
    print(
        "Missing dates:",
        history["gameDate"].isna().sum(),
    )

    assert duplicate_keys == 0
    assert history["gameDate"].notna().all()

    # --------------------------------------------------
    # Test one historical pregame roster
    # --------------------------------------------------

    prediction_date = "2025-01-15"

    prediction_timestamp = datetime(
        2025,
        1,
        15,
        18,
        0,
    )

    game_id = "PREGAME_NYK_PHI_2025_01_15"
    team = "NYK"

    roster = build_pregame_roster(
        history=history,
        team=team,
        prediction_date=prediction_date,
    )

    roster = add_recent_appearances(
        roster=roster,
        history=history,
        team=team,
        prediction_date=prediction_date,
        n_games=10,
    )

    roster = attach_expected_minutes(
        roster=roster,
        history=history,
        prediction_date=prediction_date,
    )

    roster = add_roster_status(
        roster
    )

    # --------------------------------------------------
    # Retrieve official NBA injury report
    # --------------------------------------------------

    updates = retrieve_official_injury_updates(
        report_date=datetime(
            2025,
            1,
            15,
        ),
        report_hour="05PM",
        history_df=history,
    )

    # Only use information available before prediction time
    game_updates = filter_updates_for_game(
        updates,
        team_names={
            "New York Knicks",
        },
        prediction_timestamp=prediction_timestamp
    )

    print(
        "\nOfficial NYK availability updates:",
        len(game_updates),
    )

    # --------------------------------------------------
    # Merge availability into pregame roster
    # --------------------------------------------------

    roster = apply_roster_status_adjustment(
        roster
    )

    roster = apply_availability_updates(
        roster,
        game_updates,
    )

    roster = apply_availability_adjustment(
        roster
    )

    roster = attach_pregame_player_ratings(
        roster=roster,
        ratings=ratings,
        team="NYK",
        prediction_date=prediction_date,
    )
    roster = add_cold_start_ratings(
        roster
    )

    roster["gameId"] = game_id

    roster["gameDate"] = pd.Timestamp(
        "2025-01-15"
    )

    roster["teamTricode"] = "NYK"

    baseline_team_rating = build_team_game_ratings(
        roster,
        weight_col="expectedMinutesRosterAdjusted",
    )
    print("\nBaseline NYK team rating:")
    print(
        baseline_team_rating.to_string(
            index=False
        )
    )

    injury_team_rating = build_team_game_ratings(
        roster,
        weight_col="expectedMinutesAdjusted",
    )

    comparison = baseline_team_rating.copy()

    for rating in RATING_COLS:
        comparison[
            f"{rating}Delta"
        ] = (
            injury_team_rating.iloc[0][rating]
            - baseline_team_rating.iloc[0][rating]
        )
    nyk_roster = roster
    nyk_baseline = baseline_team_rating
    nyk_injury = injury_team_rating
    print("\nInjury impact on NYK ratings:")

    print(
        comparison[
            [
                "teamTricode",
                *[
                    f"{rating}Delta"
                    for rating in RATING_COLS
                ],
            ]
        ].to_string(index=False)
    )

    print(
        "\nInjury-aware NYK team rating:"
    )

    print(
        injury_team_rating.to_string(
            index=False
        )
    )
    print(
        roster[
            [
                "personId",
                "firstName",
                "familyName",
                "availabilityStatus",
                "expectedMinutesAdjusted",
                "ratingSource",
                *RATING_COLS,
            ]
        ]
        .sort_values(
            "expectedMinutesAdjusted",
            ascending=False,
        )
        .to_string(index=False)
    )

    print(
        "\nNYK roster with availability:"
    )

    print(
        roster[
            [
                "personId",
                "firstName",
                "familyName",
                "rosterStatus",
                "expectedMinutes",
                "availabilityStatus",
                "expectedMinutesAdjusted",
                "availabilityReason",
            ]
        ]
        .sort_values(
            "expectedMinutes",
            ascending=False,
        )
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------

    assert roster["personId"].is_unique

    assert roster[
        "teamTricode"
    ].eq("NYK").all()

    assert roster[
        "expectedMinutes"
    ].notna().all()

    assert roster[
        "rosterStatus"
    ].isin(
        [
            "recent_rotation",
            "uncertain",
            "stale_candidate",
        ]
    ).all()

    print(
        "\nAvailability status counts:"
    )

    print(
        roster[
            "availabilityStatus"
        ].value_counts(
            dropna=False
        )
    )

    phi_roster = build_pregame_roster(
        history=history,
        team="PHI",
        prediction_date=prediction_date,
    )

    phi_roster = add_recent_appearances(
        roster=phi_roster,
        history=history,
        team="PHI",
        prediction_date=prediction_date,
        n_games=10,
    )

    phi_roster = attach_expected_minutes(
        roster=phi_roster,
        history=history,
        prediction_date=prediction_date,
    )

    phi_roster = add_roster_status(
        phi_roster
    )

    phi_roster = apply_roster_status_adjustment(
        phi_roster
    )

    phi_updates = filter_updates_for_game(
        updates,
        team_names={
            "Philadelphia 76ers",
        },
        prediction_timestamp=prediction_timestamp,
    )

    phi_roster = apply_availability_updates(
        phi_roster,
        phi_updates,
    )

    phi_roster = apply_availability_adjustment(
        phi_roster
    )

    phi_roster = attach_pregame_player_ratings(
        roster=phi_roster,
        ratings=ratings,
        team="PHI",
        prediction_date=prediction_date,
    )

    phi_roster = add_cold_start_ratings(
        phi_roster
    )

    phi_roster["gameId"] = game_id
    phi_roster["gameDate"] = pd.Timestamp(
        prediction_date
    )
    phi_roster["teamTricode"] = "PHI"

    phi_baseline = build_team_game_ratings(
        phi_roster,
        weight_col="expectedMinutesRosterAdjusted",
    )

    phi_injury = build_team_game_ratings(
        phi_roster,
        weight_col="expectedMinutesAdjusted",
    )
    print(
        "\nPHI injury-aware team rating:"
    )

    print(
        phi_injury.to_string(
            index=False
        )
    )
    print(
        "\nPHI roster with availability:"
    )

    print(
        phi_roster[
            [
                "personId",
                "firstName",
                "familyName",
                "rosterStatus",
                "expectedMinutes",
                "expectedMinutesRosterAdjusted",
                "availabilityStatus",
                "expectedMinutesAdjusted",
                "availabilityReason",
                "ratingSource",
            ]
        ]
        .sort_values(
            "expectedMinutes",
            ascending=False,
        )
        .to_string(index=False)
    )
    print(
        "\nPHI minutes totals:"
    )

    print(
        "Original:",
        phi_roster["expectedMinutes"].sum(),
    )

    print(
        "Roster-adjusted:",
        phi_roster[
            "expectedMinutesRosterAdjusted"
        ].sum(),
    )

    print(
        "Availability-adjusted:",
        phi_roster[
            "expectedMinutesAdjusted"
        ].sum(),
    )
    team_df = pd.concat(
        [
            nyk_injury,
            phi_injury,
        ],
        ignore_index=True,
    )
    team_df["isHome"] = (
        team_df["teamTricode"] == "PHI"
    )
    game_df = build_game_matchup_features(
        team_df
    )

    game_df = add_rating_differences(
        game_df
    )
    print(
        "\nNYK @ PHI pregame game features:"
    )

    print(
        game_df[
            [
                "gameId",
                "gameDate",
                "homeTeam",
                "awayTeam",
                "homeExpectedMinutesTotal",
                "awayExpectedMinutesTotal",
                "finishingDiff",
                "shootingDiff",
                "playmakingDiff",
                "defenseDiff",
                "reboundingDiff",
                "physicalDiff",
            ]
        ].to_string(index=False)
    )
    assert len(game_df) == 1

    assert game_df.iloc[0]["homeTeam"] == "PHI"
    assert game_df.iloc[0]["awayTeam"] == "NYK"

    diff_cols = [
        "finishingDiff",
        "shootingDiff",
        "playmakingDiff",
        "defenseDiff",
        "reboundingDiff",
        "physicalDiff",
    ]

    assert game_df[diff_cols].notna().all().all()


if __name__ == "__main__":
    main()