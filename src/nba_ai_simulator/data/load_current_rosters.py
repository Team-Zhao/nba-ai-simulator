"""Fetch and snapshot NBA season team-roster membership via nba_api.

This module intentionally DOES NOT estimate ratings/minutes, infer injuries,
or treat a season roster as an authoritative game-day active lineup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

DEFAULT_SEASON = "2026-27"
REQUIRED_API_COLUMNS = {"PLAYER_ID", "PLAYER", "TeamID"}


def resolve_team(tricode: str) -> dict:
    code = tricode.strip().upper()
    matches = [t for t in teams.get_teams() if t["abbreviation"] == code]
    if len(matches) != 1:
        raise ValueError(f"Unknown NBA team tricode: {tricode!r}")
    return matches[0]


def normalize_roster(raw: pd.DataFrame, *, team: dict, season: str, fetched_at: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError(f"{team['abbreviation']}: API returned an empty roster; do not fall back to prior-season roster")
    missing = REQUIRED_API_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"{team['abbreviation']}: missing NBA API fields {sorted(missing)}")
    ids = pd.to_numeric(raw["PLAYER_ID"], errors="coerce")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"{team['abbreviation']}: missing/duplicate PLAYER_ID")
    response_team_ids = pd.to_numeric(raw["TeamID"], errors="coerce")
    if response_team_ids.isna().any() or not response_team_ids.eq(team["id"]).all():
        raise ValueError(f"{team['abbreviation']}: returned roster contains incorrect TeamID")
    names = raw["PLAYER"].astype("string").str.strip()
    if names.isna().any() or names.eq("").any():
        raise ValueError(f"{team['abbreviation']}: invalid player name")
    out = pd.DataFrame({
        "personId": ids.astype("int64"),
        "fullName": names.astype(str),
        "teamTricode": team["abbreviation"],
        "teamId": int(team["id"]),
        "season": season,
        "rosterSource": "nba_api:CommonTeamRoster",
        "rosterFetchedAtUTC": fetched_at,
    })
    for source, target in [("POSITION", "position"), ("NUM", "jerseyNumber"), ("EXP", "experience")]:
        out[target] = raw[source].tolist() if source in raw else None
    return out.sort_values(["fullName", "personId"]).reset_index(drop=True)


def fetch_current_team_roster(tricode: str, *, season: str = DEFAULT_SEASON, timeout: int = 30) -> pd.DataFrame:
    team = resolve_team(tricode)
    # Don't silently substitute a different season if an endpoint is empty/blocked.
    endpoint = commonteamroster.CommonTeamRoster(team_id=team["id"], season=season, timeout=timeout)
    raw = endpoint.common_team_roster.get_data_frame()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return normalize_roster(raw, team=team, season=season, fetched_at=fetched_at)


def fetch_matchup_rosters(home_team: str, away_team: str, *, season: str = DEFAULT_SEASON, timeout: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if home_team.strip().upper() == away_team.strip().upper():
        raise ValueError("Home and away teams must differ")
    home = fetch_current_team_roster(home_team, season=season, timeout=timeout)
    away = fetch_current_team_roster(away_team, season=season, timeout=timeout)
    overlap = set(home.personId) & set(away.personId)
    if overlap:
        raise ValueError(f"Player IDs appear on both rosters: {sorted(overlap)}")
    return home, away


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot official NBA season roster membership")
    parser.add_argument("--home", required=True, help="Home team tricode, e.g. NYK")
    parser.add_argument("--away", required=True, help="Away team tricode, e.g. WAS")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/roster_snapshots"))
    args = parser.parse_args()
    home, away = fetch_matchup_rosters(args.home, args.away, season=args.season)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for roster in (home, away):
        team = roster.teamTricode.iloc[0]
        path = args.output_dir / f"{args.season}_{team}_{timestamp}.csv"
        roster.to_csv(path, index=False)
        print(f"{team}: {len(roster)} players -> {path}")
        print(roster[["personId", "fullName", "position"]].to_string(index=False))
    print("NOTE: Season roster != confirmed pregame active roster; check transactions, preseason participants, injuries.")


if __name__ == "__main__":
    main()
