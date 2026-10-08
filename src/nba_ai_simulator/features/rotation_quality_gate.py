"""Audit provisional live NBA rosters before sending Top-10 features to a model.

Quality status is an evidence check, not a probability calibration or performance
claim. This module does not mutate rosters and has no PyTorch dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

RATINGS = ("finishing", "shooting", "playmaking", "defense", "rebounding", "physical")
MINUTES = "expectedMinutesRosterAdjusted"
VALID_STATUSES = {"AVAILABLE", "PROBABLE", "QUESTIONABLE", "DOUBTFUL", "OUT", "UNKNOWN"}
MULTIPLIERS = {"AVAILABLE": 1., "PROBABLE": 1., "QUESTIONABLE": .5, "DOUBTFUL": .25, "OUT": 0., "UNKNOWN": 1.}


@dataclass
class RotationQualityReport:
    status: str
    team: str
    prediction_date: str
    selected_count: int
    candidate_count: int
    selected_players: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_rotation(
    roster: pd.DataFrame,
    *,
    team: str,
    prediction_date: str,
    scenario: str = "baseline",
    top_n: int = 10,
    stale_days: int = 120,
    max_fallback_minutes_share: float = 0.10,
) -> RotationQualityReport:
    """Score the *selected rotation*, not the complete season roster.

    Baseline uses untouched minutes; adjusted uses status multipliers and
    re-selects players after adjustments. UNKNOWN is never interpreted as OUT.
    A report is BLOCKED only for structural/invalid-data issues. Other concerns
    are REVIEW_REQUIRED, never silently fixed.
    """
    if scenario not in {"baseline", "injury_adjusted"}:
        raise ValueError("scenario must be baseline or injury_adjusted")
    date = pd.Timestamp(prediction_date).normalize()
    if pd.isna(date):
        raise ValueError("prediction_date is invalid")
    report = RotationQualityReport("BLOCKED", team, str(date.date()), 0, len(roster))
    required = {"personId", MINUTES, *RATINGS}
    missing = sorted(required - set(roster.columns))
    if missing:
        report.blockers.append(f"missing required columns: {missing}")
        return report
    if roster.empty or roster.personId.isna().any() or not roster.personId.is_unique:
        report.blockers.append("empty roster, missing personId, or duplicated personId")
        return report
    if "teamTricode" in roster and not roster.teamTricode.eq(team).all():
        report.blockers.append("roster teamTricode mismatch")
        return report

    f = roster.copy()
    f[MINUTES] = pd.to_numeric(f[MINUTES], errors="coerce")
    for col in RATINGS:
        f[col] = pd.to_numeric(f[col], errors="coerce")
    if not np.isfinite(f[[MINUTES, *RATINGS]].to_numpy(dtype=float)).all():
        report.blockers.append("non-finite minutes or ratings")
        return report
    if (f[MINUTES] < 0).any():
        report.blockers.append("negative expected minutes")
        return report

    f["availabilityStatus"] = f.get("availabilityStatus", pd.Series("UNKNOWN", index=f.index)).fillna("UNKNOWN").astype(str).str.upper().str.strip()
    bad = sorted(set(f.availabilityStatus) - VALID_STATUSES)
    if bad:
        report.blockers.append(f"invalid availability statuses: {bad}")
        return report
    f["auditMinutes"] = f[MINUTES]
    if scenario == "injury_adjusted":
        f["auditMinutes"] *= f.availabilityStatus.map(MULTIPLIERS)
    selected = f.loc[f.auditMinutes.gt(0)].sort_values(["auditMinutes", "personId"], ascending=[False, True], kind="stable").head(top_n).copy()
    report.selected_count = len(selected)
    if len(selected) < top_n:
        report.blockers.append(f"only {len(selected)} positive-minute players, need {top_n}")

    minute_total = float(selected.auditMinutes.sum())
    minute_fallback = selected.get("minutesSource", pd.Series("unknown", index=selected.index)).eq("fallback_20")
    rating_cold = selected.get("isColdStart", pd.Series(False, index=selected.index)).fillna(False).astype(bool)
    ratings_dates = pd.to_datetime(selected.get("ratingsAsOf", pd.Series(pd.NaT, index=selected.index)), errors="coerce")
    minutes_dates = pd.to_datetime(selected.get("minutesAsOf", pd.Series(pd.NaT, index=selected.index)), errors="coerce")
    rating_ages = (date - ratings_dates).dt.days
    minute_ages = (date - minutes_dates).dt.days
    stale_minutes = minute_ages.gt(stale_days)
    stale_ratings = rating_ages.gt(stale_days)
    fallback_share = float(selected.loc[minute_fallback, "auditMinutes"].sum() / minute_total) if minute_total > 0 else 0.
    unknown_status = selected.availabilityStatus.eq("UNKNOWN")
    if minute_fallback.any():
        report.warnings.append(f"{int(minute_fallback.sum())} selected players use unverified 20-minute fallback")
    if rating_cold.any():
        report.warnings.append(f"{int(rating_cold.sum())} selected players use cold-start ratings")
    if fallback_share > max_fallback_minutes_share:
        report.warnings.append(f"fallback minutes represent {fallback_share:.1%} of selected minutes (limit {max_fallback_minutes_share:.0%})")
    if stale_minutes.any():
        report.warnings.append(f"{int(stale_minutes.sum())} selected players have minutes older than {stale_days} days")
    if stale_ratings.any():
        report.warnings.append(f"{int(stale_ratings.sum())} selected players have ratings older than {stale_days} days")
    if unknown_status.any():
        report.warnings.append(f"{int(unknown_status.sum())} selected players have UNKNOWN availability; game-day status unverified")
    if not {"minutesSource", "minutesAsOf", "ratingsAsOf", "isColdStart"}.issubset(f.columns):
        report.warnings.append("missing provenance fields; rotation evidence incomplete")
    if minute_total <= 0:
        report.blockers.append("selected minutes total is zero")
    report.diagnostics = {
        "minutes_total": round(minute_total, 3),
        "fallback_minutes_share": round(fallback_share, 4),
        "minute_fallback_selected": int(minute_fallback.sum()),
        "rating_cold_start_selected": int(rating_cold.sum()),
        "stale_minutes_selected": int(stale_minutes.sum()),
        "stale_ratings_selected": int(stale_ratings.sum()),
        "unknown_status_selected": int(unknown_status.sum()),
        "stale_days_threshold": stale_days,
        "scenario": scenario,
    }
    report.selected_players = [
        {
            "personId": str(r.personId),
            "name": str(getattr(r, "fullName", r.personId)),
            "minutes": round(float(r.auditMinutes), 2),
            "minutesSource": str(getattr(r, "minutesSource", "unknown")),
            "availability": str(r.availabilityStatus),
        }
        for r in selected.itertuples(index=False)
    ]
    report.status = "BLOCKED" if report.blockers else "REVIEW_REQUIRED" if report.warnings else "READY"
    return report


def audit_matchup(home: pd.DataFrame, away: pd.DataFrame, *, prediction_date: str, scenarios=("baseline", "injury_adjusted")) -> dict[str, dict[str, dict]]:
    return {
        scenario: {
            "home": audit_rotation(home, team=str(home.teamTricode.iloc[0]), prediction_date=prediction_date, scenario=scenario).to_dict(),
            "away": audit_rotation(away, team=str(away.teamTricode.iloc[0]), prediction_date=prediction_date, scenario=scenario).to_dict(),
        }
        for scenario in scenarios
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Audit saved live team rosters before prediction")
    parser.add_argument("--home", required=True, help="Path to home team's live roster CSV")
    parser.add_argument("--away", required=True, help="Path to away team's live roster CSV")
    parser.add_argument("--date", required=True, help="Prediction date YYYY-MM-DD")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args()
    home = pd.read_csv(args.home)
    away = pd.read_csv(args.away)
    if "teamTricode" not in home or "teamTricode" not in away:
        parser.error("Both roster CSV files need teamTricode")
    if home.empty or away.empty:
        parser.error("Both roster CSV files must have players")
    reports = audit_matchup(home, away, prediction_date=args.date)
    for scenario, sides in reports.items():
        print(f"\n{scenario.upper()}")
        for side, r in sides.items():
            print(f"{side.upper()} {r['team']}: {r['status']} ({r['selected_count']} selected / {r['candidate_count']} candidates)")
            for label in ("blockers", "warnings"):
                for msg in r[label]:
                    print(f"  {label[:-1]}: {msg}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
