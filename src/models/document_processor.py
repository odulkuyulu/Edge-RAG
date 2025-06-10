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
from PyPDF2 import PdfReader

@dataclass
class TextChunk:
    text: str
    metadata: dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_path: str) -> List[TextChunk]:
        """Process a document and return chunks of text."""
        mime_type, _ = mimetypes.guess_type(file_path)
        text = ""

        if mime_type == "application/pdf":
            text = self._process_pdf(file_path)
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
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file using PyPDF2."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise Exception(f"Error processing PDF with PyPDF2: {e}")

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