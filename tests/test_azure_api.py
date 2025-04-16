import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure AI Language service configuration
endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
key = os.getenv("AZURE_LANGUAGE_KEY")

# Test text
text = "Microsoft's Azure AI and Google's DeepMind are developing healthcare AI systems that must address privacy concerns and bias in medical diagnosis."

# Headers
headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": key
}

# Test Entity Recognition
print("\nTesting Entity Recognition:")
entity_url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
entity_data = {
    "kind": "EntityRecognition",
    "analysisInput": {
        "documents": [
            {
                "id": "1",
                "text": text,
                "language": "en"
            }
        ]
    }
}

response = requests.post(entity_url, headers=headers, json=entity_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# Test Key Phrase Extraction
print("\nTesting Key Phrase Extraction:")
key_phrase_url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
key_phrase_data = {
    "kind": "KeyPhraseExtraction",
    "analysisInput": {
        "documents": [
            {
                "id": "1",
                "text": text,
                "language": "en"
            }
        ]
    }
}

response = requests.post(key_phrase_url, headers=headers, json=key_phrase_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# Test Sentiment Analysis
print("\nTesting Sentiment Analysis:")
sentiment_url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
sentiment_data = {
    "kind": "SentimentAnalysis",
    "analysisInput": {
        "documents": [
            {
                "id": "1",
                "text": text,
                "language": "en"
            }
        ]
    }
}

response = requests.post(sentiment_url, headers=headers, json=sentiment_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}") 