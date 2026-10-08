"""Deterministic prediction/what-if tools for the NBA assistant.

These functions reuse the frozen inference pipeline. No model fitting,
remote roster fetch, or real-world injury claims occur here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import pandas as pd

from nba_ai_simulator.models.predict_live_matchup import DEFAULT_MODEL_DIR, predict_matchup


def simulate_player_out(
    home: pd.DataFrame,
    away: pd.DataFrame,
    *,
    player_id: int,
    home_team: str,
    away_team: str,
    game_date: str,
    game_id: str = "demo",
    model_dir: Path = DEFAULT_MODEL_DIR,
    predictor: Callable = predict_matchup,
) -> dict:
    """Run an isolated hypothetical OUT scenario; no input DataFrames are mutated.

    `predictor` is injectable for offline tests; normal use calls the persisted
    PyTorch and Team Logistic artifacts from `predict_live_matchup.py`.
    """
    player_id = int(player_id)
    home_copy = home.copy(deep=True)
    away_copy = away.copy(deep=True)
    hits = []
    for side, frame in (("home", home_copy), ("away", away_copy)):
        if "personId" not in frame.columns:
            raise ValueError(f"{side} roster missing personId")
        ids = pd.to_numeric(frame["personId"], errors="raise")
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{side} roster has missing/duplicate personId")
        matching = ids.eq(player_id)
        if matching.any():
            hits.append((side, frame, matching))
    if len(hits) != 1:
        raise ValueError(f"player_id {player_id} must match exactly one roster player; found {len(hits)}")

    side, frame, matching = hits[0]
    player_name = (str(frame.loc[matching, "fullName"].iloc[0])
                   if "fullName" in frame.columns else str(player_id))

    if "availabilityStatus" not in frame.columns:
        frame["availabilityStatus"] = "UNKNOWN"
    original_status = str(frame.loc[matching, "availabilityStatus"].iloc[0])
    if original_status.upper() == "OUT":
        raise ValueError(f"player_id {player_id} is already OUT in the input; no new scenario to simulate")

    # Baseline should use original historical expected minutes, independent of
    # availability (the existing feature adapter's baseline semantics).
    # Compare to *existing evidence-adjusted* prediction before applying the
    # hypothetical OUT; do not change any verified statuses in the source rows.
    before = predictor(home_copy, away_copy, home_team=home_team,
                       away_team=away_team, game_date=game_date,
                       game_id=game_id, model_dir=model_dir)
    before_adjusted = before["predictions"]["injury_adjusted"]

    if "availabilityStatus" not in frame.columns:
        frame["availabilityStatus"] = "UNKNOWN"
    original_status = str(frame.loc[matching, "availabilityStatus"].iloc[0])
    if original_status.upper() == "OUT":
        raise ValueError(f"player_id {player_id} is already OUT in the input; no new scenario to simulate")
    frame.loc[matching, "availabilityStatus"] = "OUT"

    after = predictor(home_copy, away_copy, home_team=home_team,
                      away_team=away_team, game_date=game_date,
                      game_id=game_id, model_dir=model_dir)
    after_adjusted = after["predictions"]["injury_adjusted"]

    keys = ("team_logistic_home_win_probability", "pytorch_home_win_probability")
    deltas = {
        k + "_delta_percentage_points": round(
            100 * (float(after_adjusted[k]) - float(before_adjusted[k])), 4
        )
        for k in keys
    }
    before_ids = before_adjusted[f"{side}_top10"]
    after_ids = after_adjusted[f"{side}_top10"]
    return {
        "scenario_type": "HYPOTHETICAL_PLAYER_OUT",
        "not_real_injury_report": True,
        "game_id": game_id,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "affected_side": side,
        "player_id": player_id,
        "player_name": player_name,
        "previous_availability": original_status,
        "assumed_availability": "OUT",
        "comparison_basis": "existing evidence-adjusted scenario versus additional hypothetical OUT",
        "before": before_adjusted,
        "after": after_adjusted,
        "deltas": deltas,
        "removed_from_top10": [pid for pid in before_ids if pid not in after_ids],
        "added_to_top10": [pid for pid in after_ids if pid not in before_ids],
        "data_quality": after["data_quality"],
        "limitations": "Unverified roster/minutes. OUT excludes player; no rotation minute redistribution. No preseason validation.",
    }
