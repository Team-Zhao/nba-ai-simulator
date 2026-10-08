import pandas as pd
from nba_ai_simulator.features.rotation_quality_gate import audit_rotation

SKILLS = ["finishing", "shooting", "playmaking", "defense", "rebounding", "physical"]


def make_roster():
    rows = []
    for i in range(11):
        rows.append({"personId": 100 + i, "fullName": f"Player {i}", "teamTricode": "NYK", "expectedMinutesRosterAdjusted": 35-i, "minutesSource": "last_10_games", "isColdStart": False, "ratingsAsOf": "2026-04-12", "minutesAsOf": "2026-04-12", "availabilityStatus": "AVAILABLE", **dict.fromkeys(SKILLS, 60.)})
    return pd.DataFrame(rows)


def test_clean_ready_with_recent_data():
    r = make_roster()
    r["ratingsAsOf"] = "2026-10-07"
    r["minutesAsOf"] = "2026-10-07"
    got = audit_rotation(r, team="NYK", prediction_date="2026-10-08")
    assert got.status == "READY"
    assert got.selected_count == 10


def test_fallback_and_old_data_need_review():
    r = make_roster()
    r.loc[r.index[0], ["minutesSource", "isColdStart"]] = ["fallback_20", True]
    got = audit_rotation(r, team="NYK", prediction_date="2026-10-08")
    assert got.status == "REVIEW_REQUIRED"
    assert got.diagnostics["minute_fallback_selected"] == 1
    assert got.diagnostics["stale_minutes_selected"] == 10


def test_injury_reselects_substitute():
    r = make_roster()
    r["ratingsAsOf"] = "2026-10-07"
    r["minutesAsOf"] = "2026-10-07"
    r.loc[0, "availabilityStatus"] = "OUT"
    baseline = audit_rotation(r, team="NYK", prediction_date="2026-10-08")
    adjusted = audit_rotation(r, team="NYK", prediction_date="2026-10-08", scenario="injury_adjusted")
    assert "100" in [x["personId"] for x in baseline.selected_players]
    assert "100" not in [x["personId"] for x in adjusted.selected_players]
    assert "110" in [x["personId"] for x in adjusted.selected_players]


def test_block_invalid_minutes():
    r = make_roster()
    r.loc[0, "expectedMinutesRosterAdjusted"] = -5
    got = audit_rotation(r, team="NYK", prediction_date="2026-10-08")
    assert got.status == "BLOCKED"


def test_block_fewer_than_ten():
    r = make_roster().head(9)
    got = audit_rotation(r, team="NYK", prediction_date="2026-10-08")
    assert got.status == "BLOCKED"
