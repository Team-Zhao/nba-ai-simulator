"""Attach strictly pregame ratings and estimated minutes to current NBA season rosters.

This is an adapter, NOT a retraining step. Historical ratings are the last available
rolling pre-game ratings, which omit the player's last game's contribution. The
as-of date, age and fallback flags are exposed for honest demo diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nba_ai_simulator.features.build_pregame_rosters import load_player_history
from nba_ai_simulator.features.build_rolling_team_features import (
    RATING_COLS,
    convert_minutes_to_numeric,
)

DEFAULT_RATINGS = Path("data/processed/player_ratings_rolling.csv")
DEFAULT_SNAPSHOTS = Path("data/raw/roster_snapshots")
MINUTE_FALLBACK = 20.0  # Matches add_expected_minutes_fallback() for cold starts.


def latest_snapshot(team: str, season: str, directory: Path = DEFAULT_SNAPSHOTS) -> Path:
    candidates = sorted(directory.glob(f"{season}_{team.upper()}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No roster snapshot for {team} {season} in {directory}")
    return candidates[-1]


def attach_live_player_features(
    current_roster: pd.DataFrame,
    history: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    prediction_date: str | pd.Timestamp,
    team: str,
) -> pd.DataFrame:
    """Return ALL candidates (never top-10 here) ready for build_live_matchup_features.

    History and ratings must contain only observations from before the target date;
    this function enforces that boundary again. No target game outcomes are used.
    """
    cutoff = pd.Timestamp(prediction_date).normalize()
    if pd.isna(cutoff):
        raise ValueError("prediction_date is required")
    roster = current_roster.copy()
    required = {"personId", "teamTricode"}
    if not required.issubset(roster.columns):
        raise ValueError(f"roster missing {sorted(required - set(roster.columns))}")
    if roster.empty or roster.personId.isna().any() or not roster.personId.is_unique:
        raise ValueError("roster empty or personId invalid/duplicated")
    if not roster.teamTricode.eq(team).all():
        raise ValueError(f"team mismatch; expected {team}")
    roster["personId"] = pd.to_numeric(roster["personId"], errors="raise").astype("int64")

    for label, frame, needed in (
        ("ratings", ratings, {"personId", "gameDate", "gameId", *RATING_COLS}),
        ("history", history, {"personId", "gameDate", "gameId", "minutes"}),
    ):
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    prior_ratings = ratings.copy()
    prior_ratings["gameDate"] = pd.to_datetime(prior_ratings.gameDate)
    prior_ratings = prior_ratings.loc[prior_ratings.gameDate.lt(cutoff)].copy()
    prior_ratings = prior_ratings.sort_values(["personId", "gameDate", "gameId"])
    prior_ratings = prior_ratings.drop_duplicates("personId", keep="last")
    prior_ratings = prior_ratings.rename(columns={"gameDate": "ratingsAsOf"})
    roster = roster.merge(
        prior_ratings[["personId", "ratingsAsOf", *RATING_COLS]],
        on="personId", how="left", validate="one_to_one",
    )
    # Match original historical cold-start behavior (all six become 50 where missing).
    roster["isColdStart"] = roster[RATING_COLS].isna().any(axis=1)
    for rating in RATING_COLS:
        roster[rating] = pd.to_numeric(roster[rating], errors="coerce").fillna(50.0)
    roster["ratingSource"] = np.where(roster.isColdStart, "cold_start_50", "latest_prior_rolling")
    roster["ratingsAgeDays"] = (cutoff - roster.ratingsAsOf).dt.days

    prior_history = history.copy()
    prior_history["gameDate"] = pd.to_datetime(prior_history.gameDate)
    prior_history = prior_history.loc[prior_history.gameDate.lt(cutoff)].copy()
    prior_history["minutesNumeric"] = prior_history.minutes.map(convert_minutes_to_numeric)
    if prior_history.minutesNumeric.isna().any():
        raise ValueError("Unparseable historical minutes")
    prior_history = prior_history.sort_values(["personId", "gameDate", "gameId"])
    last_ten = prior_history.groupby("personId", sort=False).tail(10)
    minutes = last_ten.groupby("personId").agg(
        expectedMinutes=("minutesNumeric", "mean"),
        historicalMinuteGames=("minutesNumeric", "size"),
        minutesAsOf=("gameDate", "max"),
    ).reset_index()
    roster = roster.merge(minutes, on="personId", how="left", validate="one_to_one")
    roster["minutesSource"] = np.where(roster.expectedMinutes.isna(), "fallback_20", "last_10_games")
    roster["expectedMinutes"] = roster.expectedMinutes.fillna(MINUTE_FALLBACK)
    roster["expectedMinutesRosterAdjusted"] = roster.expectedMinutes.copy()
    roster["minutesAgeDays"] = (cutoff - roster.minutesAsOf).dt.days
    roster["availabilityStatus"] = "UNKNOWN"  # No injury report has been applied yet.
    if not np.isfinite(roster[[*RATING_COLS, "expectedMinutesRosterAdjusted"]].to_numpy(float)).all():
        raise ValueError("non-finite model features")
    if roster.expectedMinutesRosterAdjusted.lt(0).any():
        raise ValueError("negative minutes")
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description="Join live roster with prior-only player features")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--date", required=True, help="Prediction date YYYY-MM-DD")
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/live_rosters"))
    args = parser.parse_args()
    history = load_player_history()
    ratings = pd.read_csv(args.ratings, parse_dates=["gameDate"], dtype={"gameId": str})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for team in (args.home.upper(), args.away.upper()):
        input_path = latest_snapshot(team, args.season, args.snapshots)
        roster = pd.read_csv(input_path)
        frame = attach_live_player_features(roster, history, ratings, prediction_date=args.date, team=team)
        output = args.output_dir / f"{args.season}_{team}_{args.date}.csv"
        frame.to_csv(output, index=False)
        print(f"\n{team}: {len(frame)} candidates → {output}")
        print(f"  rating cold starts: {int(frame.isColdStart.sum())}")
        print(f"  minute fallbacks: {int(frame.minutesSource.eq('fallback_20').sum())}")
        print(f"  latest rating observation: {frame.ratingsAsOf.max()}")
        print(f"  latest minutes observation: {frame.minutesAsOf.max()}")
        print("  all availabilityStatus=UNKNOWN until verified game-day evidence is applied")
        print(frame.sort_values("expectedMinutesRosterAdjusted", ascending=False)[[
            "fullName", "expectedMinutesRosterAdjusted", "ratingSource", "ratingsAsOf",
            "minutesSource", "minutesAsOf",
        ]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
