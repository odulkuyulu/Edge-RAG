"""
Vector Store Module

Handles vector storage and similarity search using Qdrant.
"""

import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    text: str
    metadata: Dict[str, Any]
    score: float

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = QdrantClient("localhost", port=6333)
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
    
    def add_document(self, text: str, embedding: List[float], metadata: Dict[str, Any]):
        """Add a document to the vector store."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Use UUID string to ensure valid Qdrant IDs
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata
                    }
                )
            ]
        )
    
    def search(self, query_embedding: List[float], limit: int = 5) -> List[SearchResult]:
        """Search for similar documents."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            SearchResult(
                text=hit.payload["text"],
                metadata=hit.payload["metadata"],
                score=hit.score
            )
            for hit in results
        ]
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def store_embeddings(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store document chunks and their embeddings in Qdrant.
        
        Args:
            doc_id: Document identifier
            chunks: List of text chunks with metadata
            embeddings: List of embeddings corresponding to chunks
        """
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "metadata": {
                        **chunk["metadata"],
                        "doc_id": doc_id
                    }
                }
            )
            points.append(point)
            
        # Upsert points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def search_by_text(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text similarity.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        results = self.search(query_embedding=[float(x) for x in query_text.split()], limit=top_k)
        return [
            {
                "text": result.text,
                "metadata": result.metadata,
                "score": result.score
            }
            for result in results
        ] 