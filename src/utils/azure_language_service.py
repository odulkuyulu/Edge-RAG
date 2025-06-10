from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os

class AzureLanguageService:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.key = os.getenv("AZURE_LANGUAGE_KEY")
        self.text_analytics_client = None

        if self.endpoint and self.key:
            self.text_analytics_client = TextAnalyticsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            print("Azure Language client initialized.")
        else:
            print("Azure Language credentials not found. Language detection will be skipped.")

    def detect_language(self, text: str) -> str:
        if not self.text_analytics_client:
            return "en" # Default to English if service not initialized

        try:
            # The input needs to be a list of documents
            documents = [text]
            response = self.text_analytics_client.detect_language(documents, country_hint="us")
            
            # Assuming the first document is the one we're interested in for now
            if response and response[0].primary_language:
                print(f"Detected language: {response[0].primary_language.iso6391_name}")
                return response[0].primary_language.iso6391_name
            else:
                return "en" # Default to English if detection fails
        except Exception as e:
            print(f"Error detecting language with Azure Language Service: {e}")
            return "en" # Fallback to English on error 