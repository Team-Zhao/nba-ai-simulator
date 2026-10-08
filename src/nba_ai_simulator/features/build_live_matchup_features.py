"""Leakage-conscious, model-compatible matchup feature adapter (V1).

Input: CURRENT, externally validated full team rosters with six prior-only
ratings, expectedMinutesRosterAdjusted, personId, and optionally availabilityStatus.
This module deliberately does NOT claim to fetch current rosters or live injuries.

Injury-adjusted scenario: status reductions, then reselect Top 10. No minutes
redistribution in V1; use as an experimental scenario, not validated performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from nba_ai_simulator.models.player_model import (
    PLAYER_FEATURES, TOP_N_PLAYERS, get_player_feature_cols,
)

RATINGS = PLAYER_FEATURES[:-1]
MINUTES_COL = PLAYER_FEATURES[-1]
STATUS_MULTIPLIERS = {
    "OUT": 0.0,
    "DOUBTFUL": 0.25,
    "QUESTIONABLE": 0.5,
    "PROBABLE": 1.0,
    "AVAILABLE": 1.0,
    "UNKNOWN": 1.0,
}


@dataclass
class MatchupFeatures:
    baseline: pd.DataFrame
    injury_adjusted: pd.DataFrame
    baseline_rosters: dict[str, pd.DataFrame]
    adjusted_rosters: dict[str, pd.DataFrame]
    diagnostics: dict


def _prepare_full_roster(roster: pd.DataFrame, team: str) -> pd.DataFrame:
    required = ["personId", *RATINGS, MINUTES_COL]
    missing = [c for c in required if c not in roster.columns]
    if missing:
        raise ValueError(f"{team}: missing full-roster columns: {missing}")
    if roster.empty:
        raise ValueError(f"{team}: empty roster")
    if roster["personId"].isna().any() or not roster["personId"].is_unique:
        raise ValueError(f"{team}: missing or duplicate personId")
    frame = roster.copy()
    for col in [*RATINGS, MINUTES_COL]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if not np.isfinite(frame[[*RATINGS, MINUTES_COL]].to_numpy(dtype=float)).all():
        raise ValueError(f"{team}: non-finite rating or minutes; fix upstream")
    if (frame[MINUTES_COL] < 0).any():
        raise ValueError(f"{team}: negative expected minutes")
    if "availabilityStatus" not in frame:
        frame["availabilityStatus"] = "UNKNOWN"
    frame["availabilityStatus"] = frame["availabilityStatus"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    bad = set(frame["availabilityStatus"]) - set(STATUS_MULTIPLIERS)
    if bad:
        raise ValueError(f"{team}: unsupported availability statuses: {sorted(bad)}")
    return frame


def _rank_and_select(roster: pd.DataFrame, minutes_column: str) -> pd.DataFrame:
    # Do not allow OUT players to take slots from genuine rotation candidates.
    # Tie break with player id for reproducibility.
    ranked = roster.sort_values(
        [minutes_column, "personId"], ascending=[False, True], kind="stable"
    )
    selected = ranked.loc[ranked[minutes_column] > 0].head(TOP_N_PLAYERS).copy()
    return selected.reset_index(drop=True)


def _flatten(roster: pd.DataFrame, prefix: str, minutes_column: str) -> dict[str, float]:
    result = {}
    for i in range(TOP_N_PLAYERS):
        for feature in PLAYER_FEATURES:
            key = f"{prefix}_p{i + 1}_{feature}"
            if i >= len(roster):
                result[key] = 0.0
            elif feature == MINUTES_COL:
                result[key] = float(roster.iloc[i][minutes_column])
            else:
                result[key] = float(roster.iloc[i][feature])
    return result


def build_live_matchup_features(
    home_roster: pd.DataFrame,
    away_roster: pd.DataFrame,
    *,
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str,
) -> MatchupFeatures:
    """Build paired 140-feature frames from VERIFIED full pregame rosters.

    All ratings/minute estimates must be known prior to tipoff. Use provider
    roster membership, not 'team of last historical appearance', for 2026-27.
    """
    if home_team == away_team:
        raise ValueError("Home and away teams must differ")
    rosters = {
        "home": _prepare_full_roster(home_roster, home_team),
        "away": _prepare_full_roster(away_roster, away_team),
    }
    baseline_selected = {}
    adjusted_selected = {}
    baseline = {"gameId": str(game_id), "gameDate": game_date,
                "homeTeam": home_team, "awayTeam": away_team}
    adjusted = baseline.copy()
    diagnostics = {"status": "experimental_injury_scenario", "teams": {}}
    for side, frame in rosters.items():
        frame = frame.copy()
        frame["scenarioMinutes"] = (
            frame[MINUTES_COL]
            * frame["availabilityStatus"].map(STATUS_MULTIPLIERS).astype(float)
        )
        baseline_selected[side] = _rank_and_select(frame, MINUTES_COL)
        adjusted_selected[side] = _rank_and_select(frame, "scenarioMinutes")
        baseline.update(_flatten(baseline_selected[side], side, MINUTES_COL))
        adjusted.update(_flatten(adjusted_selected[side], side, "scenarioMinutes"))
        diagnostics["teams"][side] = {
            "candidates": len(frame),
            "baseline_selected": baseline_selected[side]["personId"].tolist(),
            "adjusted_selected": adjusted_selected[side]["personId"].tolist(),
            "out_count": int((frame["availabilityStatus"] == "OUT").sum()),
            "unknown_count": int((frame["availabilityStatus"] == "UNKNOWN").sum()),
            "baseline_selected_minutes_total": float(baseline_selected[side][MINUTES_COL].sum()),
            "adjusted_selected_minutes_total": float(adjusted_selected[side]["scenarioMinutes"].sum()),
        }
    b, a = pd.DataFrame([baseline]), pd.DataFrame([adjusted])
    expected = get_player_feature_cols(b.columns)
    if get_player_feature_cols(a.columns) != expected or len(expected) != 140:
        raise AssertionError("140-D model input mismatch")
    if not np.isfinite(b[expected].to_numpy(dtype=float)).all() or not np.isfinite(a[expected].to_numpy(dtype=float)).all():
        raise ValueError("Generated features have NaN or infinity")
    return MatchupFeatures(b, a, baseline_selected, adjusted_selected, diagnostics)
