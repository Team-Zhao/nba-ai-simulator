from dataclasses import dataclass
from datetime import datetime
import json
from nba_ai_simulator.llm.client import generate_text

@dataclass
class AvailabilityEvidence:
    personId: int
    team: str
    sourceType: str
    sourceUrl: str
    publishedAt: datetime
    text: str

@dataclass
class AvailabilityUpdate:
    personId: int
    team: str
    status: str
    reason: str
    confidence: str
    sourceType: str
    sourceUrl: str
    publishedAt: datetime

VALID_STATUSES = {
    "AVAILABLE",
    "PROBABLE",
    "QUESTIONABLE",
    "DOUBTFUL",
    "OUT",
    "UNKNOWN",
}

VALID_CONFIDENCE_LEVELS = {
    "high",
    "medium",
    "low",
}

def validate_availability_update(update):
    assert update.status in VALID_STATUSES, (
        f"Invalid availability status: {update.status}"
    )

    assert update.confidence in VALID_CONFIDENCE_LEVELS, (
        f"Invalid confidence level: {update.confidence}"
    )

    assert update.personId is not None
    assert update.team
    assert update.sourceType
    assert update.sourceUrl
    assert update.publishedAt is not None

    return True

def extract_update_from_evidence_rule(evidence):
    text = evidence.text.lower()

    status = "UNKNOWN"
    reason = ""
    confidence = "low"

    if "out" in text:
        status = "OUT"
        confidence = "high"

    elif "questionable" in text:
        status = "QUESTIONABLE"
        confidence = "high"

    elif "doubtful" in text:
        status = "DOUBTFUL"
        confidence = "high"

    elif "available" in text:
        status = "AVAILABLE"
        confidence = "high"

    update = AvailabilityUpdate(
        personId=evidence.personId,
        team=evidence.team,
        status=status,
        reason=reason,
        confidence=confidence,
        sourceType=evidence.sourceType,
        sourceUrl=evidence.sourceUrl,
        publishedAt=evidence.publishedAt,
    )

    validate_availability_update(update)

    return update

def build_availability_prompt(evidence):
    return f"""
    You extract NBA player availability information from a source.

    Allowed status values:
    - AVAILABLE
    - OUT
    - QUESTIONABLE
    - DOUBTFUL
    - UNKNOWN

    Allowed confidence values:
    - high
    - medium
    - low

    Rules:
    1. Use only the provided evidence.
    2. Do not infer facts that are not explicitly supported.
    3. If the evidence does not clearly state availability, return UNKNOWN.
    4. Do not convert uncertainty into certainty.
    5. Return JSON only.

    Player ID: {evidence.personId}
    Team: {evidence.team}
    Source type: {evidence.sourceType}
    Published at: {evidence.publishedAt}

    Evidence:
    {evidence.text}

    Return:
    {{
    "status": "...",
    "reason": "...",
    "confidence": "..."
    }}
    """

def extract_update_from_evidence_llm(
    evidence,
    llm_client,
):
    prompt = build_availability_prompt(evidence)

    try:
        response = llm_client(prompt)

        data = json.loads(response)

        update = AvailabilityUpdate(
            personId=evidence.personId,
            team=evidence.team,
            status=data["status"],
            reason=data.get("reason", ""),
            confidence=data["confidence"],
            sourceType=evidence.sourceType,
            sourceUrl=evidence.sourceUrl,
            publishedAt=evidence.publishedAt,
        )

        validate_availability_update(update)

        return update

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        AssertionError,
    ):
        return extract_update_from_evidence_rule(
            evidence
        )

if __name__ == "__main__":
    evidence = AvailabilityEvidence(
        personId=2544,
        team="LAL",
        sourceType="nba_official",
        sourceUrl="https://example.com",
        publishedAt=datetime(2025, 1, 15, 15, 30),
        text="Player is listed as questionable due to ankle soreness.",
    )

    update = extract_update_from_evidence_llm(
        evidence,
        generate_text,
    )


    print(update)

