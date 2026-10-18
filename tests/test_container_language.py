import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import container_language


@pytest.mark.parametrize("feature,port", [
    ("languages", 5000),
    ("entities/recognition/general", 5001),
    ("keyPhrases", 5002),
    ("sentiment", 5003),
])
def test_routes_each_feature_to_local_container(monkeypatch, feature, port):
    monkeypatch.delenv(container_language.FEATURES[feature][0], raising=False)
    payload = {"documents": [{"id": "1"}]}
    if feature == "entities/recognition/general":
        payload = {"results": payload}
    post = Mock(return_value=Mock(json=lambda: payload))
    monkeypatch.setattr(container_language.requests, "post", post)
    container_language.analyze(feature, "Microsoft", "arabic")
    args, kwargs = post.call_args
    if feature == "entities/recognition/general":
        assert args[0] == f"http://127.0.0.1:{port}/language/:analyze-text"
        assert kwargs["params"] == {"api-version": "2023-04-01"}
        assert kwargs["json"]["kind"] == "EntityRecognition"
        documents = kwargs["json"]["analysisInput"]["documents"]
    else:
        assert args[0] == f"http://127.0.0.1:{port}/text/analytics/v3.1/{feature}"
        documents = kwargs["json"]["documents"]
    assert kwargs["timeout"] == 60
    assert "Ocp-Apim-Subscription-Key" not in kwargs.get("headers", {})
    if feature != "languages":
        assert documents[0]["language"] == "ar"


@pytest.mark.parametrize("payload", [
    {"documents": [], "errors": [{"error": {"code": "UnsupportedLanguageCode"}}]},
    {"documents": []},
])
def test_document_errors_are_not_silent(monkeypatch, payload):
    monkeypatch.setattr(container_language.requests, "post", Mock(return_value=Mock(json=lambda: payload)))
    with pytest.raises(RuntimeError):
        container_language.analyze("languages", "Hello")


def test_transport_errors_do_not_fall_back_to_cloud(monkeypatch):
    post = Mock(side_effect=requests.ConnectionError("container unavailable"))
    monkeypatch.setattr(container_language.requests, "post", post)
    with pytest.raises(requests.ConnectionError):
        container_language.analyze("languages", "Hello")
    assert post.call_count == 1


def test_mode_is_opt_in(monkeypatch):
    monkeypatch.delenv("AZURE_AI_CONTAINER_MODE", raising=False)
    assert not container_language.enabled()
    monkeypatch.setenv("AZURE_AI_CONTAINER_MODE", "true")
    assert container_language.enabled()


def test_indexer_and_retriever_use_same_local_helpers(monkeypatch):
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "test-key")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "test-key")
    monkeypatch.setenv("AZURE_AI_CONTAINER_MODE", "true")
    monkeypatch.setattr("nltk.download", lambda *args, **kwargs: True)
    monkeypatch.setattr("qdrant_client.QdrantClient", Mock())
    indexer = importlib.import_module("indexer")
    retriever = importlib.import_module("retriever")
    monkeypatch.setattr(container_language, "detect_language", lambda text: {"language": "arabic", "confidence": 0.99})
    monkeypatch.setattr(container_language, "extract_entities", lambda text, language: [{"text": "Microsoft", "category": "Organization"}])
    assert indexer.detect_language("test") == "arabic"
    assert retriever.detect_language_azure("test")["language"] == "arabic"
    assert indexer.extract_entities("test", "english")[0]["text"] == "Microsoft"
    assert retriever.extract_entities_azure("test", "english") == {"Organization": ["Microsoft"]}


def test_cloud_language_detection_is_preserved(monkeypatch):
    monkeypatch.delenv("AZURE_AI_CONTAINER_MODE", raising=False)
    indexer = importlib.import_module("indexer")
    post = Mock(return_value=Mock(json=lambda: {
        "documents": [{"detectedLanguage": {"iso6391Name": "ar"}}],
    }))
    monkeypatch.setattr(indexer.requests, "post", post)
    assert indexer.detect_language("test") == "arabic"
    assert post.call_args.args[0].startswith(indexer.AZURE_LANGUAGE_ENDPOINT)
    assert post.call_args.kwargs["headers"]["Ocp-Apim-Subscription-Key"]


def test_local_pdf_failure_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_AI_CONTAINER_MODE", "true")
    indexer = importlib.import_module("indexer")
    monkeypatch.setattr(indexer, "HAS_DI", True)
    monkeypatch.setattr(indexer, "di_client", Mock(
        begin_analyze_document=Mock(side_effect=RuntimeError("local container unavailable")),
    ))
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-test")
    with pytest.raises(RuntimeError, match="local container unavailable"):
        indexer.process_with_document_intelligence(str(pdf))
