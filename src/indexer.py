"""
Indexer Module

This module handles document processing and indexing for the RAG system. It provides functionality to:
1. Process various document types (PDF, text) using Azure Document Intelligence
2. Split documents into overlapping chunks for better context preservation
3. Generate embeddings for each chunk
4. Store the chunks and their embeddings in a vector database

The module supports both PDF and text files, with special handling for PDFs using Azure's Document Intelligence service.
"""

from typing import List
import os
from dataclasses import dataclass
from pathlib import Path
import mimetypes
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from src.embeddings import TextEmbeddingModel
from src.vector_db import VectorDBClient

# Load environment variables
load_dotenv()

@dataclass
class TextChunk:
    """
    Data class representing a chunk of text with its associated metadata.
    
    Attributes:
        text (str): The actual text content of the chunk
        metadata (dict): Additional information about the chunk (source, index, etc.)
    """
    text: str
    metadata: dict

class Indexer:
    """
    A class that handles document processing, chunking, and indexing into the vector database.
    
    This class provides functionality to:
    - Process PDFs using Azure Document Intelligence
    - Process plain text files
    - Split documents into overlapping chunks
    - Generate embeddings for chunks
    - Store chunks in the vector database
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Indexer with configuration and required services.
        
        Args:
            chunk_size (int): Size of each text chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Raises:
            ValueError: If Azure Document Intelligence credentials are not found
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Azure Document Intelligence client
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
            raise ValueError("Azure Document Intelligence credentials not found. Cannot process PDFs without it.")

        # Initialize embedding generator and vector store
        self.embedding_generator = TextEmbeddingModel()
        self.vector_store = VectorDBClient()

    def process_and_index_document(self, file_path: str, file_name: str) -> dict:
        """
        Process a document, generate embeddings, and store in the vector database.
        
        Args:
            file_path (str): Path to the document file
            file_name (str): Name of the file for metadata
            
        Returns:
            dict: Processing status and statistics
            
        Raises:
            ValueError: If file type is unsupported or PDF processing is not available
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        text = ""

        # Process document based on its type
        if mime_type == "application/pdf":
            if self.doc_client:
                text = self._process_pdf_azure(file_path)
            else:
                raise ValueError("Azure Document Intelligence client not initialized. Cannot process PDF.")
        elif mime_type and mime_type.startswith("text/"):
            text = self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type or 'unknown'}")

        if not text:
            return {"message": "No text extracted from document."}

        # Split text into chunks and process each chunk
        chunks = self._chunk_text(text)
        
        # Generate embeddings and store in vector database
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_generator.generate_embedding(chunk)
            self.vector_store.add_document(
                text=chunk,
                embedding=embedding,
                metadata={
                    "source": file_name,
                    "chunk_index": i,
                    "file_type": mime_type
                }
            )
        
        return {
            "message": "Document processed and indexed successfully",
            "chunks_processed": len(chunks)
        }
    
    def _process_pdf_azure(self, file_path: str) -> str:
        """
        Extract text from a PDF using Azure Document Intelligence.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            Exception: If PDF processing fails
        """
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
            raise Exception(f"Error processing PDF with Azure Document Intelligence: {e}")

    def _process_text(self, file_path: str) -> str:
        """
        Read text from a plain text file with UTF-8 encoding, falling back to Latin-1 if needed.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: File contents as text
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks of specified size.
        
        Args:
            text (str): The text to split into chunks
            
        Returns:
            List[str]: List of text chunks with specified overlap
        """
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