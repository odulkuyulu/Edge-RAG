import logging
import os
from typing import Any, Dict, List

import requests

log = logging.getLogger(__name__)

FEATURES = {
    "languages": ("AZURE_LANGUAGE_DETECTION_ENDPOINT", "http://127.0.0.1:5000"),
    "entities/recognition/general": ("AZURE_LANGUAGE_NER_ENDPOINT", "http://127.0.0.1:5001"),
    "keyPhrases": ("AZURE_LANGUAGE_KEYPHRASE_ENDPOINT", "http://127.0.0.1:5002"),
    "sentiment": ("AZURE_LANGUAGE_SENTIMENT_ENDPOINT", "http://127.0.0.1:5003"),
}


def enabled() -> bool:
    return os.getenv("AZURE_AI_CONTAINER_MODE", "").lower() == "true"


def analyze(feature: str, text: str, language: str = "english") -> Dict[str, Any]:
    variable, default = FEATURES[feature]
    endpoint = os.getenv(variable, default).rstrip("/")
    version = os.getenv("AZURE_LANGUAGE_CONTAINER_API_VERSION", "v3.1")
    document = {"id": "1", "text": text}
    if feature != "languages":
        document["language"] = "ar" if language == "arabic" else "en"
    is_ner = feature == "entities/recognition/general"
    if is_ner:
        url = f"{endpoint}/language/:analyze-text"
        body = {"kind": "EntityRecognition", "analysisInput": {"documents": [document]}}
        params = {"api-version": "2023-04-01"}
    else:
        url = f"{endpoint}/text/analytics/{version}/{feature}"
        body = {"documents": [document]}
        params = {}
    try:
        response = requests.post(
            url,
            json=body,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if is_ner:
            payload = payload["results"]
        if payload.get("errors"):
            raise RuntimeError(f"Local Language {feature} error: {payload['errors']}")
        documents = payload.get("documents", [])
        if not documents:
            raise RuntimeError(f"Local Language {feature} returned no document result")
        return documents[0]
    except (requests.RequestException, ValueError, RuntimeError):
        log.exception("Local Language container request failed: %s", feature)
        raise


def detect_language(text: str) -> Dict[str, Any]:
    detected = analyze("languages", text)["detectedLanguage"]
    return {
        "language": "arabic" if detected["iso6391Name"] == "ar" else "english",
        "confidence": detected["confidenceScore"],
    }


def extract_entities(text: str, language: str) -> List[Dict[str, str]]:
    return [
        {"text": entity["text"], "category": entity["category"]}
        for entity in analyze("entities/recognition/general", text, language)["entities"]
    ]


def extract_key_phrases(text: str, language: str) -> List[str]:
    phrases = []
    for offset in range(0, len(text), 5000):
        phrases.extend(analyze("keyPhrases", text[offset:offset + 5000], language)["keyPhrases"])
    return phrases


def analyze_sentiment(text: str, language: str) -> Dict[str, float]:
    scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    count = 0
    for offset in range(0, len(text), 5000):
        result = analyze("sentiment", text[offset:offset + 5000], language)["confidenceScores"]
        for name in scores:
            scores[name] += result[name]
        count += 1
    if not count:
        raise ValueError("Sentiment analysis requires non-empty text")
    return {name: value / count for name, value in scores.items()}
