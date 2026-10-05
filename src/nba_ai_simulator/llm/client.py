import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"


def generate_text(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


def fake_generate_text(prompt):
    return """
    {
        "status": "QUESTIONABLE",
        "reason": "ankle soreness",
        "confidence": "high"
    }
    """

if __name__ == "__main__":
    result = generate_text(
    """
    Return JSON only:

    {
    "status": "QUESTIONABLE",
    "reason": "ankle soreness",
    "confidence": "high"
    }
    """
        )

    print(result)