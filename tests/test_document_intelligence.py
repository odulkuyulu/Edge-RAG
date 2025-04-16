import os
import pytest
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Document Analysis Client
endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
key = os.getenv("AZURE_DOC_INTEL_KEY")

@pytest.fixture
def document_client():
    """Create a Document Analysis Client fixture."""
    if not endpoint or not key:
        pytest.skip("Azure Document Intelligence credentials not found in .env file")
    return DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def test_document_analysis(document_client):
    """Test document analysis on the test PDF."""
    # Path to the test document
    test_file_path = "data/test_document.pdf"
    
    # Ensure the test file exists
    assert os.path.exists(test_file_path), f"Test file not found at {test_file_path}"
    
    # Analyze the document
    with open(test_file_path, "rb") as f:
        poller = document_client.begin_analyze_document("prebuilt-document", f)
    result = poller.result()
    
    # Basic assertions
    assert result.content, "No content extracted from the document"
    
    # Check for English text
    assert "Test Document for Document Intelligence" in result.content, "English title not found"
    assert "This is a test document" in result.content, "English content not found"
    assert "document intelligence service" in result.content.lower(), "English service reference not found"
    
    # Print detailed results
    print("\nTest Results:")
    print(f"Total content length: {len(result.content)} characters")
    print("\nExtracted Content:")
    print(result.content)
    print("\nDocument analysis completed successfully") 