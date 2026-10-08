import copy
import pandas as pd
import pytest
from nba_ai_simulator.agents.whatif_tools import simulate_player_out
from nba_ai_simulator.agents.matchup_assistant import parse_tool_decision
from nba_ai_simulator.features.build_live_matchup_features import build_live_matchup_features


def roster(start, team):
    return pd.DataFrame([{
        "personId": start + i,
        "fullName": f"Player {start+i}",
        "teamTricode": team,
        "finishing": 50+i, "shooting": 55+i, "playmaking": 52+i,
        "defense": 53+i, "rebounding": 54+i, "physical": 56+i,
        "expectedMinutesRosterAdjusted": float(35-i*2),
        "availabilityStatus": "UNKNOWN",
    } for i in range(12)])


def fake_predict(home, away, *, home_team, away_team, game_date, game_id, model_dir):
    f = build_live_matchup_features(home, away, home_team=home_team, away_team=away_team,
                                    game_date=game_date, game_id=game_id)
    results = {}
    for label, frame, picks in (("baseline", f.baseline, f.baseline_rosters),
                                ("injury_adjusted", f.injury_adjusted, f.adjusted_rosters)):
        # deterministic stand-in for model in this unit test (not a real NBA probability)
        ph = float(frame["home_p1_finishing"].iloc[0] / 100)
        results[label] = {
            "team_logistic_home_win_probability": ph,
            "pytorch_home_win_probability": ph,
            "home_top10": picks["home"]["personId"].astype(int).tolist(),
            "away_top10": picks["away"]["personId"].astype(int).tolist(),
        }
    return {"predictions": results, "data_quality": "UNVERIFIED"}


def test_top11_bench_enters_and_inputs_unchanged():
    home, away = roster(100, "NYK"), roster(200, "WAS")
    snapshot = home.copy(deep=True)
    result = simulate_player_out(home, away, player_id=100, home_team="NYK",
                                 away_team="WAS", game_date="2026-10-08", predictor=fake_predict)
    assert 100 in result["before"]["home_top10"]
    assert 100 not in result["after"]["home_top10"]
    assert 110 in result["added_to_top10"]
    assert result["scenario_type"] == "HYPOTHETICAL_PLAYER_OUT"
    pd.testing.assert_frame_equal(home, snapshot)


def test_unknown_player_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        simulate_player_out(roster(100,"NYK"), roster(200,"WAS"), player_id=999,
                            home_team="NYK", away_team="WAS", game_date="2026-10-08", predictor=fake_predict)


def test_already_out_rejected():
    home = roster(100,"NYK")
    home.loc[0, "availabilityStatus"] = "OUT"
    with pytest.raises(ValueError, match="already OUT"):
        simulate_player_out(home, roster(200,"WAS"), player_id=100,
                            home_team="NYK", away_team="WAS", game_date="2026-10-08", predictor=fake_predict)


def test_strict_llm_commands():
    assert parse_tool_decision('{"tool":"simulate_player_out","player_id":1628973}')["player_id"] == 1628973
    with pytest.raises(ValueError):
        parse_tool_decision('{"tool":"unknown"}')
    with pytest.raises(ValueError):
        parse_tool_decision('{"tool":"simulate_player_out","player_id":1628973,"probability":1}')
