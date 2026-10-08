"""Run with: uv run pytest -q tests/test_live_matchup_features.py"""
import pandas as pd
import pytest

from nba_ai_simulator.features.build_live_matchup_features import build_live_matchup_features
from nba_ai_simulator.models.player_model import get_player_feature_cols, pregame_arrays


def roster(first_id=100, injured=False):
    rows = []
    for i in range(12):
        row = {
            "personId": first_id + i,
            "finishing": 55 + i, "shooting": 65 + i,
            "playmaking": 60 + i, "defense": 50 + i,
            "rebounding": 55 + i, "physical": 45 + i,
            "expectedMinutesRosterAdjusted": float(35-i*2),
            "availabilityStatus": "OUT" if (injured and i == 0) else "AVAILABLE",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_injured_top_player_replaced_by_bench():
    result = build_live_matchup_features(
        roster(100, injured=True), roster(200),
        game_id="test_game", game_date="2026-10-08", home_team="NYK", away_team="BOS",
    )
    baseline_ids = result.diagnostics["teams"]["home"]["baseline_selected"]
    adjusted_ids = result.diagnostics["teams"]["home"]["adjusted_selected"]
    assert 100 in baseline_ids and 100 not in adjusted_ids
    assert 110 not in baseline_ids and 110 in adjusted_ids
    cols = get_player_feature_cols(result.baseline.columns)
    assert len(cols) == 140
    raw, minutes = pregame_arrays(result.injury_adjusted, cols)
    assert raw.shape == (1, 140) and minutes.shape == (1, 2, 10)
    assert result.injury_adjusted["home_p10_expectedMinutesRosterAdjusted"].iloc[0] == 15.0


def test_invalid_minutes_fail_closed():
    home = roster()
    home.loc[0, "expectedMinutesRosterAdjusted"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_live_matchup_features(home, roster(200), game_id="x", game_date="2026-10-08", home_team="NYK", away_team="BOS")


def test_unknown_status_fail_closed():
    home = roster()
    home.loc[0, "availabilityStatus"] = "MAYBE"
    with pytest.raises(ValueError, match="unsupported"):
        build_live_matchup_features(home, roster(200), game_id="x", game_date="2026-10-08", home_team="NYK", away_team="BOS")
