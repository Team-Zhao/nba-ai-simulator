import pandas as pd
import pytest

from nba_ai_simulator.features.attach_live_player_features import attach_live_player_features

SKILLS = ["finishing", "shooting", "playmaking", "defense", "rebounding", "physical"]


def fixture_frames():
    roster = pd.DataFrame({"personId": [10, 11, 12], "teamTricode": ["NYK"] * 3,
                           "fullName": ["A", "B", "New Player"]})
    ratings = pd.DataFrame([
        {"personId": 10, "gameDate": "2026-04-01", "gameId": "001", **dict.fromkeys(SKILLS, 80)},
        {"personId": 10, "gameDate": "2026-10-09", "gameId": "002", **dict.fromkeys(SKILLS, 100)},
        {"personId": 11, "gameDate": "2026-04-01", "gameId": "003", **dict.fromkeys(SKILLS, 60)},
    ])
    history = pd.DataFrame([
        {"personId": 10, "gameDate": "2026-03-01", "gameId": "001", "minutes": "20:00"},
        {"personId": 10, "gameDate": "2026-03-02", "gameId": "002", "minutes": "30:00"},
        {"personId": 10, "gameDate": "2026-10-09", "gameId": "003", "minutes": "40:00"},
        {"personId": 11, "gameDate": "2026-03-01", "gameId": "004", "minutes": "10:00"},
    ])
    return roster, history, ratings


def test_strict_cutoff_and_cold_start():
    roster, history, ratings = fixture_frames()
    got = attach_live_player_features(roster, history, ratings, prediction_date="2026-10-08", team="NYK")
    a = got.set_index("personId")
    assert len(got) == 3
    assert a.loc[10, "finishing"] == 80
    assert a.loc[10, "expectedMinutesRosterAdjusted"] == 25
    assert a.loc[12, "finishing"] == 50
    assert a.loc[12, "expectedMinutesRosterAdjusted"] == 20
    assert a.loc[12, "isColdStart"]
    assert (got.availabilityStatus == "UNKNOWN").all()


def test_team_mismatch_rejected():
    roster, history, ratings = fixture_frames()
    with pytest.raises(ValueError, match="team mismatch"):
        attach_live_player_features(roster, history, ratings, prediction_date="2026-10-08", team="WAS")
