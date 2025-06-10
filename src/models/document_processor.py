"""
Document Processor Module

Handles document processing including:
- File type detection
- Text extraction
- Document chunking
"""

from typing import List
import os
from dataclasses import dataclass
from pathlib import Path
import mimetypes
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TextChunk:
    text: str
    metadata: dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.azure_doc_intel_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.azure_doc_intel_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if self.azure_doc_intel_endpoint and self.azure_doc_intel_key:
            self.doc_client = DocumentAnalysisClient(
                endpoint=self.azure_doc_intel_endpoint,
                credential=AzureKeyCredential(self.azure_doc_intel_key)
            )
            print("Azure Document Intelligence client initialized.")
        else:
            self.doc_client = None
            raise ValueError("Azure Document Intelligence credentials not found. Cannot process PDFs without it.") # Removed PyPDF2 fallback

    def process_document(self, file_path: str) -> List[TextChunk]:
        """Process a document and return chunks of text."""
        mime_type, _ = mimetypes.guess_type(file_path)
        text = ""

        if mime_type == "application/pdf":
            if self.doc_client:
                text = self._process_pdf_azure(file_path)
            else:
                raise ValueError("Azure Document Intelligence client not initialized. Cannot process PDF.") # Removed PyPDF2 fallback
        elif mime_type and mime_type.startswith("text/"):
            text = self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type or 'unknown'}")

        if not text:
            return []

        chunks = self._chunk_text(text)
        
        return [
            TextChunk(
                text=chunk,
                metadata={
                    "source": os.path.basename(file_path),
                    "chunk_index": i,
                    "file_type": mime_type
                }
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _process_pdf_azure(self, file_path: str) -> str:
        """Extract text from a PDF using Azure Document Intelligence."""
        print(f"Processing PDF '{file_path}' using Azure Document Intelligence.")
        try:
            with open(file_path, "rb") as f:
                poller = self.doc_client.begin_analyze_document(
                    "prebuilt-layout", f
                )
            result = poller.result()
            return result.content
        except Exception as e:
            print(f"Error processing PDF with Azure Document Intelligence: {e}.")
            raise Exception(f"Error processing PDF with Azure Document Intelligence: {e}") # Removed PyPDF2 fallback

    # Removed _process_pdf_pypdf2 method

    def _process_text(self, file_path: str) -> str:
        """Read text from a plain text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of text
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Add chunk to list
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
        
        return chunks 