"""Auditable NBA prediction explanations, optional Ollama-controlled framing.

Input: JSON emitted by predict_live_matchup or matchup_assistant.
The model can select a wording emphasis, never supply statistical facts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import requests

PROB_KEYS = ("team_logistic_home_win_probability", "pytorch_home_win_probability")
MODEL_LABELS = ("Team Logistic", "PyTorch")


def _prob(value: Any) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("Invalid win probability")
    return value


def _extract(payload: dict) -> tuple[dict, dict | None]:
    data = payload.get("tool_result", payload)
    if "before" in data and "after" in data and data.get("scenario_type") == "HYPOTHETICAL_PLAYER_OUT":
        return data, {"before": data["before"], "after": data["after"]}
    if "predictions" in data and "baseline" in data["predictions"]:
        return data, None
    raise ValueError("Expected a saved prediction or player-out tool result")


def build_facts(payload: dict) -> dict:
    data, scenario = _extract(payload)
    home, away = str(data["home_team"]), str(data["away_team"])
    if not home or not away or home == away:
        raise ValueError("Invalid teams")
    if scenario:
        b, a = scenario["before"], scenario["after"]
        before, after = [_prob(b[k]) for k in PROB_KEYS], [_prob(a[k]) for k in PROB_KEYS]
        deltas = [100 * (n - o) for o, n in zip(before, after)]
        if deltas[0] * deltas[1] < 0:
            observation = "models_disagree"
        elif all(abs(d) < 0.005 for d in deltas):
            observation = "almost_no_change"
        else:
            observation = "models_similar_direction"
        return {
            "kind": "player_out", "home_team": home, "away_team": away,
            "game_date": str(data["game_date"]), "observation": observation,
            "player_name": str(data["player_name"]), "player_id": int(data["player_id"]),
            "before": before, "after": after, "delta_pp": deltas,
            "added_ids": [int(x) for x in data.get("added_to_top10", [])],
            "removed_ids": [int(x) for x in data.get("removed_from_top10", [])],
        }
    baseline = data["predictions"]["baseline"]
    probs = [_prob(baseline[k]) for k in PROB_KEYS]
    winners = [home if p >= 0.5 else away for p in probs]
    return {
        "kind": "prediction", "home_team": home, "away_team": away,
        "game_date": str(data["game_date"]), "observation": "same_winner" if winners[0] == winners[1] else "different_winner",
        "probs": probs, "winners": winners,
    }


# Fixed, verified language. LLM only chooses among these predefined emphases.
FOCUS = {
    "models_disagree": {
        "model_difference": "The models respond in opposite directions. This is a model-sensitivity difference, not evidence that the absent player improves the team.",
        "simulation_limits": "This is a conditional scenario, not a causal estimate of player value; teammates' minutes are not redistributed.",
    },
    "models_similar_direction": {
        "model_difference": "Both models move in the same general direction, but the size of the change depends on the model.",
        "simulation_limits": "This is a conditional scenario, not a causal estimate of player value; teammates' minutes are not redistributed.",
    },
    "almost_no_change": {
        "model_difference": "Neither model shows a material change at the precision displayed; this does not establish that the player has no impact.",
        "simulation_limits": "The model does not redistribute teammates' minutes after the hypothetical absence.",
    },
    "same_winner": {
        "agreement": "The two models select the same winner, but this agreement does not validate the game-day lineup or calibrate these preseason probabilities.",
        "data_limits": "The projected rotation and availability have not been verified for game day.",
    },
    "different_winner": {
        "agreement": "The two models select different winners; treat this as model disagreement rather than a definitive result.",
        "data_limits": "The projected rotation and availability have not been verified for game day.",
    },
}


def _choose_focus(facts: dict, llm: Callable[[str], str] | None) -> tuple[str, str]:
    options = FOCUS[facts["observation"]]
    default = next(iter(options))
    if llm is None:
        return default, "deterministic"
    prompt = (
        "Choose an explanation emphasis. Return ONLY a JSON object with one key 'focus' "
        "whose value is exactly one of the permitted strings. No other keys or prose. "
        "Do not introduce facts, players, teams, numbers, or injury claims.\n"
        f"Observation: {facts['observation']}\n"
        f"Permitted focus values: {json.dumps(list(options))}\n"
        "JSON:"
    )
    try:
        raw = llm(prompt)
        decision = json.loads(raw.strip())
        if isinstance(decision, dict) and set(decision) == {"focus"} and decision["focus"] in options:
            return decision["focus"], "ollama_validated"
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    except Exception:
        # Network or local model failure: never compromise prediction presentation.
        pass
    return default, "deterministic_fallback"


def explain(payload: dict, llm: Callable[[str], str] | None = None) -> dict:
    facts = build_facts(payload)
    focus, source = _choose_focus(facts, llm)
    home, away = facts["home_team"], facts["away_team"]
    parts = [f"{away} @ {home} ({facts['game_date']})."]
    if facts["kind"] == "player_out":
        parts.append(f"HYPOTHETICAL ONLY: {facts['player_name']} (ID {facts['player_id']}) is assumed OUT; this is not a real injury report.")
        for label, before, after, delta in zip(MODEL_LABELS, facts["before"], facts["after"], facts["delta_pp"]):
            parts.append(f"{label}: {home} win probability {before:.2%} → {after:.2%} ({delta:+.2f} percentage points).")
        parts.append(f"Top-10 changes: removed {facts['removed_ids']}; added {facts['added_ids']}.")
        parts.append(FOCUS[facts["observation"]][focus])
        parts.append("Experimental scenario. No rotation-minutes redistribution; historical ratings and projected minutes may be stale.")
    else:
        for label, p, winner in zip(MODEL_LABELS, facts["probs"], facts["winners"]):
            parts.append(f"{label}: {home} win probability {p:.2%}; predicted winner {winner}.")
        parts.append(FOCUS[facts["observation"]][focus])
        parts.append("Experimental prediction. Game-day roster and availability are unverified; historical ratings and projected minutes may be stale.")
    return {"explanation": " ".join(parts), "focus": focus, "focus_source": source, "observation": facts["observation"], "verified_facts": facts}


def ollama_generate(prompt: str) -> str:
    from nba_ai_simulator.llm.client import OLLAMA_URL, OLLAMA_MODEL
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "format": "json", "think": False,
        "options": {"temperature": 0, "num_predict": 60},
    }, timeout=45)
    response.raise_for_status()
    answer = response.json().get("response", "")
    if not isinstance(answer, str):
        raise ValueError("Ollama response must be a string")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded explanation of saved NBA prediction or hypothetical result")
    parser.add_argument("--input", required=True, type=Path, help="JSON file from predict_live_matchup or matchup_assistant")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic narrative only")
    parser.add_argument("--output", type=Path, help="Optional output JSON")
    args = parser.parse_args()
    result = explain(json.loads(args.input.read_text()), llm=None if args.no_llm else ollama_generate)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
