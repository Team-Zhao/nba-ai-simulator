import pytest
from nba_ai_simulator.agents.explanation_layer import explain, build_facts


def sample_whatif():
    return {"tool_result": {
        "scenario_type": "HYPOTHETICAL_PLAYER_OUT", "home_team": "NYK", "away_team": "WAS", "game_date": "2026-10-08",
        "player_name": "Jalen Brunson", "player_id": 1628973,
        "before": {"team_logistic_home_win_probability": 0.7316411137580872, "pytorch_home_win_probability": 0.7514511346817017},
        "after": {"team_logistic_home_win_probability": 0.7377572655677795, "pytorch_home_win_probability": 0.7459855675697327},
        "removed_from_top10": [1628973], "added_to_top10": [1630164],
    }}


def test_disagreement_and_exact_numbers():
    out = explain(sample_whatif())
    assert out['observation'] == 'models_disagree'
    assert '73.16% → 73.78% (+0.61 percentage points)' in out['explanation']
    assert '75.15% → 74.60% (-0.55 percentage points)' in out['explanation']
    assert 'not a real injury report' in out['explanation']


def test_llm_cannot_inject_arbitrary_content():
    out = explain(sample_whatif(), llm=lambda _: '{"focus":"model_difference", "opponent":"Boston Celtics"}')
    assert out['focus_source'] == 'deterministic_fallback'
    assert 'Boston Celtics' not in out['explanation']


def test_llm_valid_choice_changes_only_safe_language():
    out = explain(sample_whatif(), llm=lambda _: '{"focus":"simulation_limits"}')
    assert out['focus_source'] == 'ollama_validated'
    assert out['focus'] == 'simulation_limits'
    assert '73.16%' in out['explanation']


def test_prediction_and_wrapped_input():
    payload = {"predictions": {"baseline": {"team_logistic_home_win_probability": .73, "pytorch_home_win_probability": .75}},
               "game_date": "2026-10-08", "home_team": "NYK", "away_team": "WAS"}
    out = explain(payload)
    assert '73.00%' in out['explanation']
    assert 'predicted winner NYK' in out['explanation']


def test_reject_invalid_probability():
    bad = sample_whatif()
    bad['tool_result']['after']['pytorch_home_win_probability'] = 1.5
    with pytest.raises(ValueError):
        build_facts(bad)


def test_llm_error_fallback():
    def broken(_):
        raise TimeoutError('offline')
    assert explain(sample_whatif(), llm=broken)['focus_source'] == 'deterministic_fallback'
