"""Minimal LLM command router for local Ollama: choose a tool, never invent probabilities.

Run with --no-llm for deterministic local smoke testing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd
import requests
from nba_ai_simulator.llm.client import OLLAMA_URL, OLLAMA_MODEL
from nba_ai_simulator.models.predict_live_matchup import predict_matchup
from nba_ai_simulator.agents.whatif_tools import simulate_player_out


def parse_tool_decision(raw: str) -> dict:
    """Fail closed if the LLM returns anything besides a compact JSON command."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        raise ValueError("Ollama returned an empty response for tool routing")
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw[:300]!r}") from exc
    if not isinstance(decision, dict) or decision.get("tool") not in ("predict_matchup", "simulate_player_out"):
        raise ValueError("Unsupported tool decision")
    if decision["tool"] == "simulate_player_out":
        if set(decision) != {"tool", "player_id"}:
            raise ValueError("Unexpected arguments")
        if isinstance(decision["player_id"], bool) or not isinstance(decision["player_id"], int):
            raise ValueError("player_id must be integer")
    elif set(decision) != {"tool"}:
        raise ValueError("Unexpected arguments")
    return decision


def route_explicit_player_out(question: str, home: pd.DataFrame, away: pd.DataFrame) -> dict | None:
    """Route an explicit named-player absence scenario without trusting LLM-generated facts.

    A surname is permitted only if it identifies exactly one player across both
    provided rosters. Ambiguity fails closed rather than choosing an arbitrary ID.
    """
    q = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    absence_patterns = (
        r"\bout\b", r"\bcan t play\b", r"\bcannot play\b",
        r"\bdoesn t play\b", r"\bwon t play\b", r"\bnot playing\b",
        r"\bmiss(?:es|ing)?\b", r"\bwithout\b", r"\bsidelined\b",
        r"\bruled out\b", r"\binjur(?:ed|y)\b",
    )
    if not any(re.search(pat, q) for pat in absence_patterns):
        return None
    roster = pd.concat([home, away], ignore_index=True)
    if not {"fullName", "personId"}.issubset(roster.columns):
        raise ValueError("Roster must include fullName and personId for safe routing")
    matches = []
    for _, row in roster.iterrows():
        name = re.sub(r"[^a-z0-9]+", " ", str(row["fullName"]).casefold()).strip()
        last = name.split()[-1] if name else ""
        if (name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q)) or (
            len(last) >= 4 and re.search(r"(?<!\w)" + re.escape(last) + r"(?!\w)", q)
        ):
            matches.append(int(row["personId"]))
    unique = set(matches)
    if len(unique) == 1:
        return {"tool": "simulate_player_out", "player_id": unique.pop()}
    if len(unique) > 1:
        raise ValueError("Ambiguous player name in hypothetical OUT request")
    # Never downgrade a hypothetical absence question to a plain prediction.
    raise ValueError("Could not identify a unique roster player for hypothetical OUT request")


def choose_tool(question: str, home: pd.DataFrame, away: pd.DataFrame) -> dict:
    explicit = route_explicit_player_out(question, home, away)
    if explicit is not None:
        return explicit
    # Only allow player IDs present in these two rosters, not arbitrary entities.
    names = []
    for side, df in (("home", home), ("away", away)):
        for _, row in df.iterrows():
            names.append({"personId": int(row["personId"]), "fullName": str(row.get("fullName", "")), "team": side})
    prompt = (
        "You route an NBA user request to one of two local tools. Return a single JSON object ONLY. "
        "For the current matchup request, use {\"tool\":\"predict_matchup\"}. "
        "For a hypothetical player OUT request, use {\"tool\":\"simulate_player_out\",\"player_id\":INTEGER}. "
        "Player must match the provided roster, otherwise return predict_matchup. "
        "Do not claim an injury is real. No probabilistic predictions from yourself.\n"
        f"Roster IDs/names: {json.dumps(names, ensure_ascii=False)}\n"
        f"User request: {question}\nJSON:"
    )
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_predict": 160},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload.get("response", "")
    if not isinstance(raw, str):
        raise ValueError(f"Ollama response field must be text, got {type(raw).__name__}")
    decision = parse_tool_decision(raw)
    if decision["tool"] == "simulate_player_out":
        roster_ids = set(pd.to_numeric(pd.concat([home["personId"], away["personId"]])).astype(int))
        if decision["player_id"] not in roster_ids:
            raise ValueError("Model selected player_id absent from matchup rosters")
    return decision


def run_assistant(*, question: str, home: pd.DataFrame, away: pd.DataFrame,
                  home_team: str, away_team: str, game_date: str,
                  game_id: str = "demo", decision: dict | None = None) -> dict:
    decision = parse_tool_decision(json.dumps(decision)) if decision is not None else choose_tool(question, home, away)
    if decision["tool"] == "predict_matchup":
        data = predict_matchup(home, away, home_team=home_team, away_team=away_team,
                               game_date=game_date, game_id=game_id)
    else:
        data = simulate_player_out(home, away, player_id=decision["player_id"],
                                   home_team=home_team, away_team=away_team,
                                   game_date=game_date, game_id=game_id)
    if decision["tool"] == "predict_matchup":
        p = data["predictions"]["baseline"]
        answer = (
            f"{away_team} @ {home_team}: Team Logistic estimates "
            f"{100*p['team_logistic_home_win_probability']:.2f}% home win probability; "
            f"PyTorch estimates {100*p['pytorch_home_win_probability']:.2f}%. "
            "Lineups and injury statuses are unverified; this is an experimental preseason estimate."
        )
    else:
        b, a = data["before"], data["after"]
        answer = (
            f"HYPOTHETICAL only: if {data['player_name']} (ID {data['player_id']}) "
            f"is OUT, Team Logistic home win probability changes "
            f"from {100*b['team_logistic_home_win_probability']:.2f}% "
            f"to {100*a['team_logistic_home_win_probability']:.2f}%; "
            f"PyTorch changes from {100*b['pytorch_home_win_probability']:.2f}% "
            f"to {100*a['pytorch_home_win_probability']:.2f}%. "
            f"Added to top 10: {data['added_to_top10']}. "
            "This assumes no minutes redistribution and is not a real injury report."
        )
    return {"question": question, "selected_tool": decision, "assistant_answer": answer,
            "tool_result": data}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--away", type=Path, required=True)
    p.add_argument("--home-team", required=True)
    p.add_argument("--away-team", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--no-llm", action="store_true", help="Bypass local Ollama for smoke tests")
    p.add_argument("--player-out", type=int, help="With --no-llm, simulate this playerId")
    args = p.parse_args()
    if args.player_out is not None and not args.no_llm:
        p.error("--player-out requires --no-llm")
    decision = ({"tool": "simulate_player_out", "player_id": args.player_out}
                if args.no_llm and args.player_out is not None else
                {"tool": "predict_matchup"} if args.no_llm else None)
    out = run_assistant(question=args.question, home=pd.read_csv(args.home), away=pd.read_csv(args.away),
                        home_team=args.home_team, away_team=args.away_team,
                        game_date=args.date, decision=decision)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
