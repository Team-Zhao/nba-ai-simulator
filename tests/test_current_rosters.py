import pandas as pd
import pytest

from nba_ai_simulator.data.load_current_rosters import normalize_roster, fetch_matchup_rosters

TEAM = {"id": 1610612752, "abbreviation": "NYK"}


def test_normalization_preserves_ids_and_source():
    raw = pd.DataFrame({"PLAYER_ID": [2, 1], "PLAYER": ["Second", "First"], "TeamID": [TEAM["id"]]*2, "POSITION": ["G", "F"]})
    out = normalize_roster(raw, team=TEAM, season="2026-27", fetched_at="2026-10-07T00:00:00+00:00")
    assert out["personId"].tolist() == [1, 2]
    assert out["teamTricode"].tolist() == ["NYK", "NYK"]
    assert out["rosterSource"].iloc[0] == "nba_api:CommonTeamRoster"


def test_invalid_team_membership_rejected():
    raw = pd.DataFrame({"PLAYER_ID": [1], "PLAYER": ["First"], "TeamID": [0]})
    with pytest.raises(ValueError, match="incorrect TeamID"):
        normalize_roster(raw, team=TEAM, season="2026-27", fetched_at="now")


def test_empty_roster_rejected():
    with pytest.raises(ValueError, match="empty roster"):
        normalize_roster(pd.DataFrame(), team=TEAM, season="2026-27", fetched_at="now")


def test_duplicate_ids_rejected():
    raw = pd.DataFrame({"PLAYER_ID": [1, 1], "PLAYER": ["First", "First"], "TeamID": [TEAM["id"]]*2})
    with pytest.raises(ValueError, match="duplicate"):
        normalize_roster(raw, team=TEAM, season="2026-27", fetched_at="now")
