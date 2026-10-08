import pandas as pd
from nba_ai_simulator.features.build_pregame_rosters import (
    load_player_history,
    build_pregame_roster,
    add_recent_appearances,
    attach_expected_minutes,
    add_roster_status,
    apply_roster_status_adjustment,
    attach_pregame_player_ratings,
    RATING_COLS,
    add_cold_start_ratings,
)
import requests
from datetime import datetime
from nba_ai_simulator.retrieval.injury_report_retriever import (
    retrieve_official_injury_updates,
    filter_updates_for_game,
    apply_availability_updates,
)
from datetime import timedelta
from zoneinfo import ZoneInfo
from nba_api.stats.endpoints import (
    scheduleleaguev2,
)
from nba_ai_simulator.features.build_pregame_rosters import (
    apply_availability_adjustment,
)


TOP_N_PLAYERS = 10

PLAYER_FEATURES = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
    "expectedMinutesRosterAdjusted",
    "expectedMinutesAdjusted",
    "isOut",
    "isDoubtful",
    "isQuestionable",
    "isProbable",
]

NBA_TEAM_TRICODES = {
    "ATL", "BOS", "BKN", "CHA", "CHI",
    "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM",
    "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "TOR", "UTA", "WAS",
}

def select_top_players(
    roster_df,
    top_n=TOP_N_PLAYERS,
):
    roster = roster_df.copy()

    roster = (
        roster
        .sort_values(
            "expectedMinutesRosterAdjusted",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return roster


def flatten_roster(
    roster_df,
    prefix,
    top_n=TOP_N_PLAYERS,
):
    roster = select_top_players(
        roster_df,
        top_n=top_n,
    )

    features = {}

    for slot in range(top_n):
        slot_number = slot + 1

        if slot < len(roster):
            player = roster.iloc[slot]

            for feature in PLAYER_FEATURES:
                features[
                    f"{prefix}_p{slot_number}_{feature}"
                ] = player[feature]

        else:
            for feature in PLAYER_FEATURES:
                features[
                    f"{prefix}_p{slot_number}_{feature}"
                ] = 0.0

    return features


def build_player_level_game_row(
    home_roster,
    away_roster,
    game_id,
    game_date,
    home_team,
    away_team,
):
    row = {
        "gameId": game_id,
        "gameDate": game_date,
        "homeTeam": home_team,
        "awayTeam": away_team,
    }

    home_features = flatten_roster(
        home_roster,
        prefix="home",
    )

    away_features = flatten_roster(
        away_roster,
        prefix="away",
    )

    row.update(home_features)
    row.update(away_features)

    return row

def load_schedule_with_cutoffs(
    season="2025-26",
):
    endpoint = (
        scheduleleaguev2
        .ScheduleLeagueV2(
            season=season
        )
    )

    schedule = (
        endpoint
        .get_data_frames()[0]
        .copy()
    )

    schedule["gameId"] = (
        schedule["gameId"]
        .astype(str)
        .str.zfill(10)
    )

    schedule = schedule[
        schedule[
            "homeTeam_teamTricode"
        ].isin(
            NBA_TEAM_TRICODES
        )
        & schedule[
            "awayTeam_teamTricode"
        ].isin(
            NBA_TEAM_TRICODES
        )
    ].copy()

    schedule[
        "tipoffUTC"
    ] = pd.to_datetime(
        schedule[
            "gameDateTimeUTC"
        ],
        utc=True,
    )

    eastern = ZoneInfo(
        "America/New_York"
    )

    schedule[
        "tipoffET"
    ] = (
        schedule["tipoffUTC"]
        .dt.tz_convert(
            eastern
        )
    )

    schedule[
        "predictionTimestamp"
    ] = (
        schedule["tipoffET"]
        - timedelta(
            minutes=30
        )
    )

    # Current injury-report retriever
    # expects naive Eastern time.
    schedule[
        "predictionTimestamp"
    ] = (
        schedule[
            "predictionTimestamp"
        ]
        .dt.tz_localize(None)
    )

    schedule[
        "gameDate"
    ] = (
        schedule[
            "tipoffET"
        ]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    schedule[
        "homeTeam"
    ] = schedule[
        "homeTeam_teamTricode"
    ]

    schedule[
        "awayTeam"
    ] = schedule[
        "awayTeam_teamTricode"
    ]

    schedule[
        "homeTeamName"
    ] = (
        schedule[
            "homeTeam_teamCity"
        ]
        .astype(str)
        .str.cat(
            schedule[
                "homeTeam_teamName"
            ].astype(str),
            sep=" ",
        )
    )

    schedule[
        "awayTeamName"
    ] = (
        schedule[
            "awayTeam_teamCity"
        ]
        .astype(str)
        .str.cat(
            schedule[
                "awayTeam_teamName"
            ].astype(str),
            sep=" ",
        )
    )

    return schedule[
        [
            "gameId",
            "gameDate",
            "homeTeam",
            "awayTeam",
            "homeTeamName",
            "awayTeamName",
            "tipoffET",
            "predictionTimestamp",
        ]
    ].copy()

def build_injury_aware_player_dataset(
    history,
    ratings,
    games,
    schedule,
    max_games=None,
):
    games = games.copy()

    games["gameId"] = (
        games["gameId"]
        .astype(str)
        .str.zfill(10)
    )

    merged = games.merge(
        schedule,
        on="gameId",
        how="inner",
        suffixes=(
            "",
            "_schedule",
        ),
        validate="one_to_one",
    )

    merged = merged[
        merged[
            "gameDate"
        ]
        >= pd.Timestamp(
            "2026-01-30"
        )
    ].copy()

    merged = (
        merged
        .sort_values(
            [
                "gameDate",
                "predictionTimestamp",
                "gameId",
            ]
        )
        .reset_index(drop=True)
    )

    if max_games is not None:
        merged = (
            merged
            .head(max_games)
            .copy()
        )

    rows = []

    total_games = len(
        merged
    )

    for index, game in (
        merged.iterrows()
    ):
        game_id = (
            game["gameId"]
        )

        game_date = pd.Timestamp(
            game["gameDate"]
        )

        prediction_timestamp = (
            game[
                "predictionTimestamp"
            ]
        )

        home_team = (
            game["homeTeam"]
        )

        away_team = (
            game["awayTeam"]
        )

        home_team_name = (
            game["homeTeamName"]
        )

        away_team_name = (
            game["awayTeamName"]
        )

        print()
        print(
            f"[{index + 1}/"
            f"{total_games}] "
            f"{away_team} @ "
            f"{home_team}"
        )

        print(
            "Prediction cutoff:",
            prediction_timestamp,
        )

        try:
            updates = (
                retrieve_official_injury_updates(
                    prediction_timestamp=(
                        prediction_timestamp
                    ),
                    history_df=history,
                )
            )

        except FileNotFoundError:
            print(
                "No injury report "
                "- skipping game"
            )

            continue

        (
            row,
            home_roster,
            away_roster,
        ) = (
            build_injury_aware_player_level_game_row(
                history=history,
                ratings=ratings,
                updates=updates,
                game_id=game_id,
                game_date=game_date,
                prediction_timestamp=(
                    prediction_timestamp
                ),
                home_team=home_team,
                home_team_name=(
                    home_team_name
                ),
                away_team=away_team,
                away_team_name=(
                    away_team_name
                ),
            )
        )

        row[
            "homePointDiff"
        ] = game[
            "homePointDiff"
        ]

        row[
            "homeWin"
        ] = int(
            game["homeWin"]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )

def build_historical_pregame_roster(
    history,
    ratings,
    team,
    prediction_date,
):
    # 1. Reconstruct roster using only prior games
    roster = build_pregame_roster(
        history=history,
        team=team,
        prediction_date=prediction_date,
    )

    # 2. Count appearances in previous 10 team games
    roster = add_recent_appearances(
        roster=roster,
        history=history,
        team=team,
        prediction_date=prediction_date,
        n_games=10,
    )

    # 3. Estimate expected minutes
    roster = attach_expected_minutes(
        roster=roster,
        history=history,
        prediction_date=prediction_date,
    )

    # 4. Label roster status
    roster = add_roster_status(
        roster
    )

    # 5. Zero out stale candidates
    roster = apply_roster_status_adjustment(
        roster
    )

    # 6. Attach strictly-prior player ratings
    roster = (
        attach_strictly_prior_player_ratings(
            roster=roster,
            ratings=ratings,
            prediction_date=prediction_date,
        )
    )

    return roster

def build_player_level_dataset(
    history,
    ratings,
    games,
    max_games=None,
):
    games = (
        games
        .sort_values(
            [
                "gameDate",
                "gameId",
            ]
        )
        .reset_index(drop=True)
    )

    if max_games is not None:
        games = (
            games
            .head(max_games)
            .copy()
        )

    rows = []

    total_games = len(games)

    for index, game in (
        games.iterrows()
    ):
        game_id = str(
            game["gameId"]
        ).zfill(10)

        game_date = pd.Timestamp(
            game["gameDate"]
        )

        home_team = (
            game["homeTeam"]
        )

        away_team = (
            game["awayTeam"]
        )

        print(
            f"[{index + 1}/{total_games}] "
            f"{away_team} @ {home_team} "
            f"{game_date.date()}"
        )

        home_roster = (
            build_historical_pregame_roster(
                history=history,
                ratings=ratings,
                team=home_team,
                prediction_date=game_date,
            )
        )

        away_roster = (
            build_historical_pregame_roster(
                history=history,
                ratings=ratings,
                team=away_team,
                prediction_date=game_date,
            )
        )

        row = (
            build_player_level_game_row(
                home_roster=home_roster,
                away_roster=away_roster,
                game_id=game_id,
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
            )
        )

        row["homePointDiff"] = (
            game["homePointDiff"]
        )

        row["homeWin"] = int(
            game["homeWin"]
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )

def attach_strictly_prior_player_ratings(
    roster,
    ratings,
    prediction_date,
):
    roster = roster.copy()
    ratings = ratings.copy()

    prediction_date = pd.Timestamp(
        prediction_date
    )

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
                "gameId",
            ]
        )
        .drop_duplicates(
            subset=["personId"],
            keep="last",
        )
    )

    prior = prior[
        [
            "personId",
            *RATING_COLS,
        ]
    ]

    roster = roster.merge(
        prior,
        on="personId",
        how="left",
        validate="one_to_one",
    )

    roster["ratingSource"] = (
        "strictly_prior"
    )

    roster = add_cold_start_ratings(
        roster
    )

    return roster

def add_historical_availability(
    roster,
    updates,
    team_name,
    prediction_timestamp,
):
    roster = roster.copy()

    team_updates = filter_updates_for_game(
        updates,
        team_names={
            team_name,
        },
        prediction_timestamp=prediction_timestamp,
    )

    roster = apply_availability_updates(
        roster,
        team_updates,
    )

    roster = apply_availability_adjustment(
        roster
    )

    roster["isOut"] = (
        roster["availabilityStatus"]
        .eq("OUT")
        .astype(int)
    )

    roster["isDoubtful"] = (
        roster["availabilityStatus"]
        .eq("DOUBTFUL")
        .astype(int)
    )

    roster["isQuestionable"] = (
        roster["availabilityStatus"]
        .eq("QUESTIONABLE")
        .astype(int)
    )

    roster["isProbable"] = (
        roster["availabilityStatus"]
        .eq("PROBABLE")
        .astype(int)
    )

    return roster

def build_injury_aware_player_level_game_row(
    history,
    ratings,
    updates,
    game_id,
    game_date,
    prediction_timestamp,
    home_team,
    home_team_name,
    away_team,
    away_team_name,
):
    home_roster = (
        build_historical_pregame_roster(
            history=history,
            ratings=ratings,
            team=home_team,
            prediction_date=game_date,
        )
    )

    away_roster = (
        build_historical_pregame_roster(
            history=history,
            ratings=ratings,
            team=away_team,
            prediction_date=game_date,
        )
    )

    home_roster = (
        add_historical_availability(
            roster=home_roster,
            updates=updates,
            team_name=home_team_name,
            prediction_timestamp=prediction_timestamp,
        )
    )

    away_roster = (
        add_historical_availability(
            roster=away_roster,
            updates=updates,
            team_name=away_team_name,
            prediction_timestamp=prediction_timestamp,
        )
    )

    row = build_player_level_game_row(
        home_roster=home_roster,
        away_roster=away_roster,
        game_id=game_id,
        game_date=game_date,
        home_team=home_team,
        away_team=away_team,
    )

    return (
        row,
        home_roster,
        away_roster,
    )

def audit_injury_report_coverage(
    history,
    schedule,
    sample_size=10,
):
    schedule = (
        schedule
        .sort_values(
            [
                "gameDate",
                "predictionTimestamp",
                "gameId",
            ]
        )
        .reset_index(drop=True)
    )

    if len(schedule) <= sample_size:
        sample = schedule.copy()

    else:
        sample = (
            schedule
            .sample(
                n=sample_size,
                random_state=42,
            )
            .sort_values(
                [
                    "gameDate",
                    "predictionTimestamp",
                    "gameId",
                ]
            )
            .reset_index(drop=True)
        )

    results = []

    for _, game in sample.iterrows():
        prediction_timestamp = (
            game["predictionTimestamp"]
        )

        print(
            game["gameDate"].date(),
            game["awayTeam"],
            "@",
            game["homeTeam"],
            "-",
            prediction_timestamp,
            end=": ",
        )

        try:
            retrieve_official_injury_updates(
                prediction_timestamp=(
                    prediction_timestamp
                ),
                history_df=history,
            )

            print("FOUND")

            status = "FOUND"

        except FileNotFoundError:
            print("MISSING")

            status = "MISSING"

        except requests.exceptions.RequestException as exc:
            print(
                "NETWORK_ERROR",
                type(exc).__name__,
            )

            status = "NETWORK_ERROR"

        results.append(
            {
                "gameId": game["gameId"],
                "gameDate": game["gameDate"],
                "predictionTimestamp": (
                    prediction_timestamp
                ),
                "status": status,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    print()
    print(
        result_df["status"]
        .value_counts(
            dropna=False
        )
    )

    found_count = (
        result_df["status"]
        .eq("FOUND")
        .sum()
    )

    missing_count = (
        result_df["status"]
        .eq("MISSING")
        .sum()
    )

    network_error_count = (
        result_df["status"]
        .eq("NETWORK_ERROR")
        .sum()
    )

    print()

    print(
        "FOUND:",
        found_count,
    )

    print(
        "MISSING:",
        missing_count,
    )

    print(
        "NETWORK_ERROR:",
        network_error_count,
    )

    print(
        "Confirmed coverage:",
        f"{found_count / len(result_df):.1%}",
    )

    return result_df

def audit_injury_report_coverage_by_period(
    history,
    schedule,
):
    periods = {
        "early": (
            "2025-10-22",
            "2025-11-30",
        ),
        "middle": (
            "2025-12-01",
            "2026-01-29",
        ),
        "holdout": (
            "2026-01-30",
            "2026-04-12",
        ),
    }

    all_results = []

    for period_name, (
        start_date,
        end_date,
    ) in periods.items():
        print()
        print(
            "=============================="
        )

        print(
            f"Period: {period_name}"
        )

        print(
            start_date,
            "to",
            end_date,
        )

        print(
            "=============================="
        )

        period_schedule = schedule[
            (
                schedule["gameDate"]
                >= pd.Timestamp(
                    start_date
                )
            )
            &
            (
                schedule["gameDate"]
                <= pd.Timestamp(
                    end_date
                )
            )
        ].copy()

        result = (
            audit_injury_report_coverage(
                history=history,
                schedule=period_schedule,
                sample_size=10,
            )
        )

        result["period"] = (
            period_name
        )

        all_results.append(
            result
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    print()
    print(
        "=============================="
    )

    print(
        "Coverage summary"
    )

    print(
        "=============================="
    )

    summary = (
        combined
        .groupby(
            [
                "period",
                "status",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        summary
    )

    return combined

if __name__ == "__main__":
    history = (
        load_player_history()
    )

    schedule = (
        load_schedule_with_cutoffs(
            season="2025-26"
        )
    )

    coverage_df = (
        audit_injury_report_coverage_by_period(
            history=history,
            schedule=schedule,
        )
    )