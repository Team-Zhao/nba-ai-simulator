from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from nba_api.stats.endpoints import (
    scheduleleaguev2,
)

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    mean_squared_error,
)

from nba_ai_simulator.features.build_pregame_rosters import (
    load_player_history,
    build_pregame_team_rating,
)

from nba_ai_simulator.features.build_rolling_game_features import (
    build_game_matchup_features,
    add_rating_differences,
)

from nba_ai_simulator.models.predict_game import (
    predict_from_game_features,
)

from nba_ai_simulator.retrieval.injury_report_retriever import (
    retrieve_official_injury_updates,
)


EASTERN = ZoneInfo(
    "America/New_York"
)

HOLDOUT_START = pd.Timestamp(
    "2026-01-30"
)

NBA_TEAM_TRICODES = {
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
}

N_GAMES = 200


def load_schedule():
    schedule = (
        scheduleleaguev2
        .ScheduleLeagueV2(
            season="2025-26",
        )
    )

    schedule_df = (
        schedule.get_data_frames()[0]
    )

    schedule_df[
        "gameDateTimeUTC"
    ] = pd.to_datetime(
        schedule_df[
            "gameDateTimeUTC"
        ],
        utc=True,
    )

    schedule_df[
        "gameDate"
    ] = (
        schedule_df[
            "gameDateTimeUTC"
        ]
        .dt.tz_convert(
            EASTERN
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    return schedule_df


def build_prediction_timestamp(
    game_datetime_utc,
):
    game_datetime_utc = pd.Timestamp(
        game_datetime_utc
    )

    if (
        game_datetime_utc.tzinfo
        is None
    ):
        game_datetime_utc = (
            game_datetime_utc
            .tz_localize(
                timezone.utc
            )
        )

    game_datetime_et = (
        game_datetime_utc
        .tz_convert(
            EASTERN
        )
    )

    prediction_timestamp = (
        game_datetime_et
        - timedelta(
            minutes=30
        )
    )

    return (
        prediction_timestamp
        .tz_localize(None)
        .to_pydatetime()
    )


def main():
    # --------------------------------------------------
    # Load historical player data
    # --------------------------------------------------

    history = load_player_history()

    ratings = pd.read_csv(
        "data/processed/"
        "player_ratings_rolling.csv",
        parse_dates=["gameDate"],
        dtype={"gameId": str},
    )

    ratings["gameId"] = (
        ratings["gameId"]
        .str.zfill(10)
    )

    # --------------------------------------------------
    # Load original rolling baseline features
    # --------------------------------------------------

    baseline_df = pd.read_csv(
        "data/processed/"
        "game_features_rolling.csv",
        parse_dates=["gameDate"],
        dtype={"gameId": str},
    )

    baseline_df["gameId"] = (
        baseline_df["gameId"]
        .str.zfill(10)
    )

    # --------------------------------------------------
    # Load NBA schedule
    # --------------------------------------------------

    schedule_df = load_schedule()
    test_games = (
        schedule_df[
            (
                schedule_df["gameDate"]
                >= HOLDOUT_START
            )
            & (
                schedule_df[
                    "gameStatusText"
                ]
                == "Final"
            )
            & (
                schedule_df[
                    "homeTeam_teamTricode"
                ]
                .isin(
                    NBA_TEAM_TRICODES
                )
            )
            & (
                schedule_df[
                    "awayTeam_teamTricode"
                ]
                .isin(
                    NBA_TEAM_TRICODES
                )
            )
        ]
        .sort_values(
            "gameDateTimeUTC"
        )
        .head(N_GAMES)
        .copy()
    )

    results = []

    # --------------------------------------------------
    # Backtest games
    # --------------------------------------------------

    for _, game in (
        test_games.iterrows()
    ):
        game_id = str(
            game["gameId"]
        ).zfill(10)

        home_team = (
            game[
                "homeTeam_teamTricode"
            ]
        )

        away_team = (
            game[
                "awayTeam_teamTricode"
            ]
        )

        home_team_name = (
            str(
                game[
                    "homeTeam_teamCity"
                ]
            )
            + " "
            + str(
                game[
                    "homeTeam_teamName"
                ]
            )
        )

        away_team_name = (
            str(
                game[
                    "awayTeam_teamCity"
                ]
            )
            + " "
            + str(
                game[
                    "awayTeam_teamName"
                ]
            )
        )

        prediction_timestamp = (
            build_prediction_timestamp(
                game[
                    "gameDateTimeUTC"
                ]
            )
        )

        prediction_date = (
            prediction_timestamp
            .date()
            .isoformat()
        )

        print()
        print(
            f"Backtesting "
            f"{away_team} @ "
            f"{home_team}"
        )

        print(
            "Prediction cutoff:",
            prediction_timestamp,
        )

        # --------------------------------------------------
        # Injury report
        # --------------------------------------------------

        try:
            updates = (
                retrieve_official_injury_updates(
                    prediction_timestamp=(
                        prediction_timestamp
                    ),
                    history_df=history,
                )
            )
        except FileNotFoundError as exc:
            print(
                "Skipping game because "
                "no injury report was found:"
            )
            print(exc)
            continue

        # --------------------------------------------------
        # Home team pregame reconstruction
        # --------------------------------------------------

        (
            home_roster,
            home_baseline,
            home_injury,
        ) = build_pregame_team_rating(
            history=history,
            ratings=ratings,
            updates=updates,
            team=home_team,
            team_name=home_team_name,
            prediction_date=(
                prediction_date
            ),
            prediction_timestamp=(
                prediction_timestamp
            ),
            game_id=game_id,
        )

        # --------------------------------------------------
        # Away team pregame reconstruction
        # --------------------------------------------------

        (
            away_roster,
            away_baseline,
            away_injury,
        ) = build_pregame_team_rating(
            history=history,
            ratings=ratings,
            updates=updates,
            team=away_team,
            team_name=away_team_name,
            prediction_date=(
                prediction_date
            ),
            prediction_timestamp=(
                prediction_timestamp
            ),
            game_id=game_id,
        )

        # --------------------------------------------------
        # Injury-aware game features
        # --------------------------------------------------

        team_df = pd.concat(
            [
                home_injury,
                away_injury,
            ],
            ignore_index=True,
        )

        team_df["isHome"] = (
            team_df[
                "teamTricode"
            ]
            == home_team
        )

        injury_game_df = (
            build_game_matchup_features(
                team_df
            )
        )

        injury_game_df = (
            add_rating_differences(
                injury_game_df
            )
        )

        injury_prediction = (
            predict_from_game_features(
                injury_game_df
            )
        )

        # --------------------------------------------------
        # Original rolling baseline prediction
        # --------------------------------------------------

        baseline_game_df = (
            baseline_df[
                baseline_df[
                    "gameId"
                ]
                == game_id
            ]
            .copy()
        )

        if (
            len(baseline_game_df)
            != 1
        ):
            raise ValueError(
                f"Expected exactly one "
                f"baseline row for "
                f"game {game_id}, "
                f"found "
                f"{len(baseline_game_df)}."
            )

        baseline_prediction = (
            predict_from_game_features(
                baseline_game_df
            )
        )

        # --------------------------------------------------
        # Actual outcome
        # --------------------------------------------------

        actual_home_score = int(
            game[
                "homeTeam_score"
            ]
        )

        actual_away_score = int(
            game[
                "awayTeam_score"
            ]
        )

        actual_margin = (
            actual_home_score
            - actual_away_score
        )

        actual_home_win = (
            actual_margin > 0
        )

        # --------------------------------------------------
        # Save comparison
        # --------------------------------------------------

        results.append(
            {
                "gameId": game_id,
                "gameDate": (
                    prediction_date
                ),
                "awayTeam": away_team,
                "homeTeam": home_team,

                "actualMargin": (
                    actual_margin
                ),
                "actualHomeWin": (
                    actual_home_win
                ),

                # ------------------------------
                # Rolling baseline
                # ------------------------------

                "baselinePredictedMargin": (
                    baseline_prediction[
                        "predictedMargin"
                    ]
                ),

                "baselineLogisticHomeWinProbability": (
                    baseline_prediction[
                        "logisticHomeWinProbability"
                    ]
                ),

                "baselineXgboostHomeWinProbability": (
                    baseline_prediction[
                        "xgboostHomeWinProbability"
                    ]
                ),

                "baselinePredictedHomeWin": (
                    baseline_prediction[
                        "predictedHomeWin"
                    ]
                ),

                "baselineCorrectWinner": (
                    actual_home_win
                    == baseline_prediction[
                        "predictedHomeWin"
                    ]
                ),

                "baselineMarginError": abs(
                    actual_margin
                    - baseline_prediction[
                        "predictedMargin"
                    ]
                ),

                # ------------------------------
                # Injury-aware
                # ------------------------------

                "injuryPredictedMargin": (
                    injury_prediction[
                        "predictedMargin"
                    ]
                ),

                "injuryLogisticHomeWinProbability": (
                    injury_prediction[
                        "logisticHomeWinProbability"
                    ]
                ),

                "injuryXgboostHomeWinProbability": (
                    injury_prediction[
                        "xgboostHomeWinProbability"
                    ]
                ),

                "injuryPredictedHomeWin": (
                    injury_prediction[
                        "predictedHomeWin"
                    ]
                ),

                "injuryCorrectWinner": (
                    actual_home_win
                    == injury_prediction[
                        "predictedHomeWin"
                    ]
                ),

                "injuryMarginError": abs(
                    actual_margin
                    - injury_prediction[
                        "predictedMargin"
                    ]
                ),
            }
        )

    # --------------------------------------------------
    # Results dataframe
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print(
        "\nBatch backtest results:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Actual labels
    # --------------------------------------------------

    actual_home_wins = (
        results_df[
            "actualHomeWin"
        ]
        .astype(int)
    )

    actual_margins = (
        results_df[
            "actualMargin"
        ]
    )

    # --------------------------------------------------
    # Rolling baseline metrics
    # --------------------------------------------------

    baseline_accuracy = (
        results_df[
            "baselineCorrectWinner"
        ]
        .mean()
    )

    baseline_log_loss = log_loss(
        actual_home_wins,
        results_df[
            "baselineLogisticHomeWinProbability"
        ],
    )

    baseline_brier = (
        brier_score_loss(
            actual_home_wins,
            results_df[
                "baselineLogisticHomeWinProbability"
            ],
        )
    )

    baseline_mae = (
        results_df[
            "baselineMarginError"
        ]
        .mean()
    )

    baseline_rmse = (
        mean_squared_error(
            actual_margins,
            results_df[
                "baselinePredictedMargin"
            ],
        )
        ** 0.5
    )

    # --------------------------------------------------
    # Injury-aware metrics
    # --------------------------------------------------

    injury_accuracy = (
        results_df[
            "injuryCorrectWinner"
        ]
        .mean()
    )

    injury_log_loss = log_loss(
        actual_home_wins,
        results_df[
            "injuryLogisticHomeWinProbability"
        ],
    )

    injury_brier = (
        brier_score_loss(
            actual_home_wins,
            results_df[
                "injuryLogisticHomeWinProbability"
            ],
        )
    )

    injury_mae = (
        results_df[
            "injuryMarginError"
        ]
        .mean()
    )

    injury_rmse = (
        mean_squared_error(
            actual_margins,
            results_df[
                "injuryPredictedMargin"
            ],
        )
        ** 0.5
    )

    # --------------------------------------------------
    # XGBoost accuracy
    # --------------------------------------------------

    baseline_xgb_predictions = (
        results_df[
            "baselineXgboostHomeWinProbability"
        ]
        >= 0.5
    )

    injury_xgb_predictions = (
        results_df[
            "injuryXgboostHomeWinProbability"
        ]
        >= 0.5
    )

    baseline_xgb_accuracy = (
        (
            baseline_xgb_predictions
            == results_df[
                "actualHomeWin"
            ]
        )
        .mean()
    )

    injury_xgb_accuracy = (
        (
            injury_xgb_predictions
            == results_df[
                "actualHomeWin"
            ]
        )
        .mean()
    )

    baseline_xgb_log_loss = (
        log_loss(
            actual_home_wins,
            results_df[
                "baselineXgboostHomeWinProbability"
            ],
        )
    )

    injury_xgb_log_loss = (
        log_loss(
            actual_home_wins,
            results_df[
                "injuryXgboostHomeWinProbability"
            ],
        )
    )

    baseline_xgb_brier = (
        brier_score_loss(
            actual_home_wins,
            results_df[
                "baselineXgboostHomeWinProbability"
            ],
        )
    )

    injury_xgb_brier = (
        brier_score_loss(
            actual_home_wins,
            results_df[
                "injuryXgboostHomeWinProbability"
            ],
        )
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        "Games tested:",
        len(results_df),
    )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Logistic Winner Accuracy",
                "Logistic Log Loss",
                "Logistic Brier Score",
                "XGBoost Winner Accuracy",
                "XGBoost Log Loss",
                "XGBoost Brier Score",
                "Linear Margin MAE",
                "Linear Margin RMSE",
            ],
            "Rolling Baseline": [
                baseline_accuracy,
                baseline_log_loss,
                baseline_brier,
                baseline_xgb_accuracy,
                baseline_xgb_log_loss,
                baseline_xgb_brier,
                baseline_mae,
                baseline_rmse,
            ],
            "Injury Aware": [
                injury_accuracy,
                injury_log_loss,
                injury_brier,
                injury_xgb_accuracy,
                injury_xgb_log_loss,
                injury_xgb_brier,
                injury_mae,
                injury_rmse,
            ],
        }
    )

    comparison[
        "Difference"
    ] = (
        comparison[
            "Injury Aware"
        ]
        - comparison[
            "Rolling Baseline"
        ]
    )

    print()
    print(
        "Baseline vs Injury-Aware"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df.to_csv(
        "data/processed/"
        "pregame_backtest_results.csv",
        index=False,
    )

    comparison.to_csv(
        "data/processed/"
        "pregame_backtest_comparison.csv",
        index=False,
    )

    print()
    print(
        "Saved results to:"
    )

    print(
        "data/processed/"
        "pregame_backtest_results.csv"
    )

    print(
        "data/processed/"
        "pregame_backtest_comparison.csv"
    )


if __name__ == "__main__":
    main()