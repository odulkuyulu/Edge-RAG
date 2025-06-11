"""
Embedding Model Module

This module provides functionality for generating text embeddings using Ollama's local models.
It handles:
1. Single text embedding generation
2. Batch embedding generation for multiple text chunks
3. Embedding size normalization
4. Error handling and fallback mechanisms

The module uses the BGE-M3 model by default, which provides high-quality embeddings
suitable for semantic search and similarity matching.
"""

import os
import ollama
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class TextEmbeddingModel:
    """
    A class for generating text embeddings using Ollama's local models.
    
    This class provides methods for:
    - Generating embeddings for individual text inputs
    - Batch processing multiple text chunks
    - Normalizing embedding sizes
    - Handling API errors gracefully
    """
    
    def __init__(self, model_name: str = "bge-m3", base_url: str = "http://localhost:11434"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the Ollama model to use
            base_url (str): Base URL for the Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_size = int(os.getenv("EMBEDDING_SIZE", "1024"))
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        
        Args:
            text (str): Input text to generate embedding for
            
        Returns:
            List[float]: Vector embedding of the input text
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Call Ollama API to generate embedding
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            
            # Normalize embedding size to match expected dimensions
            if len(embedding) < self.embedding_size:
                # Pad with zeros if embedding is too short
                embedding = np.pad(embedding, (0, self.embedding_size - len(embedding)), 'constant', constant_values=0)
            elif len(embedding) > self.embedding_size:
                # Truncate if embedding is too long
                embedding = embedding[:self.embedding_size]
                
            return embedding
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
            
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks.
        
        This method processes each chunk individually and handles failures gracefully
        by using zero vectors for failed embeddings.
        
        Args:
            chunks (List[Dict[str, Any]]): List of text chunks with metadata
            
        Returns:
            List[List[float]]: List of embeddings corresponding to each chunk
        """
        embeddings = []
        for chunk in chunks:
            try:
                embedding = self.generate_embedding(chunk["text"])
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Use zero vector as fallback for failed embeddings
                    embeddings.append([0.0] * self.embedding_size)
            except Exception:
                # Use zero vector as fallback for failed embeddings
                embeddings.append([0.0] * self.embedding_size)
                
        return embeddings 