"""
Embedding Model Module

Handles text embedding generation using Ollama's local models.
"""

import os
import ollama
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class EmbeddingModel:
    def __init__(self, model_name: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_size = int(os.getenv("EMBEDDING_SIZE", "1024"))
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            
            # Ensure embedding size matches expected size
            if len(embedding) < self.embedding_size:
                embedding = np.pad(embedding, (0, self.embedding_size - len(embedding)), 'constant', constant_values=0)
            elif len(embedding) > self.embedding_size:
                embedding = embedding[:self.embedding_size]
                
            return embedding
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
            
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            List of embeddings corresponding to each chunk
        """
        embeddings = []
        for chunk in chunks:
            embedding = self.generate_embedding(chunk["text"])
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # If embedding generation fails, use zero vector
                embeddings.append([0.0] * self.embedding_size)
                
        return embeddings 