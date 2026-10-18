"""Run on the Docker host to verify local inference before indexing."""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")
os.environ["AZURE_AI_CONTAINER_MODE"] = "true"
os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = "http://127.0.0.1:5010"
os.environ["AZURE_DOCUMENT_INTELLIGENCE_API_VERSION"] = "2022-08-31"
os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11435"

import container_language
from indexer import process_with_document_intelligence


def main() -> None:
    text = "Microsoft invested in a partnership with G42 in the United Arab Emirates."
    assert container_language.detect_language(text)["language"] == "english"
    print("Local Language detection: PASS")
    entities = container_language.extract_entities(text, "english")
    assert entities
    print("Local NER: PASS")
    assert container_language.extract_key_phrases(text, "english")
    print("Local key phrases: PASS")
    sentiment = container_language.analyze_sentiment(text, "english")
    assert set(sentiment) == {"positive", "neutral", "negative"}
    print("Local sentiment: PASS")
    pdf = process_with_document_intelligence(str(ROOT / "data" / "eventss.pdf"))
    assert len(pdf["text"].strip()) > 10
    assert pdf["metadata"]["method"] == "azure_di"
    print("Local Document Intelligence PDF analysis: PASS")
    response = requests.get("http://127.0.0.1:6333/healthz", timeout=15)
    response.raise_for_status()
    print("Local Qdrant: PASS")
    response = requests.get("http://127.0.0.1:11435/api/tags", timeout=15)
    response.raise_for_status()
    names = {model["name"] for model in response.json()["models"]}
    assert {"bge-m3:latest", "qwen2.5:0.5b"} <= names, names
    print("Local Ollama models: PASS")
    response = requests.post(
        "http://127.0.0.1:11435/api/embeddings",
        json={"model": "bge-m3", "prompt": "passage: Local inference verification"},
        timeout=180,
    )
    response.raise_for_status()
    assert len(response.json()["embedding"]) == 1024
    print("Local Ollama 1024-dimensional embedding: PASS")
    response = requests.post(
        "http://127.0.0.1:11435/api/generate",
        json={"model": "qwen2.5:0.5b", "prompt": "Reply only READY", "stream": False,
              "options": {"num_predict": 16}},
        timeout=180,
    )
    response.raise_for_status()
    assert response.json()["response"].strip()
    print("Local Ollama text generation: PASS")


def verify_rag() -> None:
    from qdrant_client import QdrantClient
    from streamlit.testing.v1 import AppTest

    client = QdrantClient(url="http://127.0.0.1:6333")
    points, _ = client.scroll("rag_docs_en_bge-m3", limit=100, with_payload=True)
    assert len(points) == 8, len(points)
    sources = set()
    for point in points:
        assert point.payload is not None
        sources.add(point.payload["source"])
    assert sources == {"ai_en.txt", "eventss.pdf"}
    print("Local indexed English documents: PASS (8 chunks)")
    app = AppTest.from_file(str(ROOT / "src" / "enhanced_app.py"), default_timeout=180).run()
    assert not app.exception, [str(error) for error in app.exception]
    assert any("Local Docker demo" in message.value for message in app.info)
    app.text_area[0].input("What is the Microsoft G42 partnership about?")
    search = next(button for button in app.button if "Enhanced Search & Generate" in button.label)
    search.click().run(timeout=180)
    assert not app.exception, [str(error) for error in app.exception]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert int(metrics["Sources Found"]) > 0, metrics
    assert int(metrics["Sources Used"]) > 0, metrics
    print("Local Streamlit retrieval and generation flow: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true", help="Also test the indexed Streamlit query flow")
    args = parser.parse_args()
    main()
    if args.rag:
        verify_rag()
