"""
Backend API for Edge RAG Application

This module provides the FastAPI endpoints for:
- Document upload and processing
- Query handling and response generation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import shutil
from datetime import datetime
import uuid

from ..models.document_processor import DocumentProcessor
from ..models.embedding_model import EmbeddingModel
from ..models.llm_model import LLMModel
from ..utils.vector_store import VectorStore

app = FastAPI(title="Edge RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
embedding_model = EmbeddingModel()
llm_model = LLMModel()
vector_store = VectorStore()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    Returns the document ID and processing status.
    """
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = f"data/uploads/{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        chunks = document_processor.process(file_path)
        
        # Generate embeddings
        embeddings = embedding_model.generate_embeddings(chunks)
        
        # Store in vector database
        vector_store.store_embeddings(doc_id, chunks, embeddings)
        
        return {
            "status": "success",
            "document_id": doc_id,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: str, top_k: int = 3):
    """
    Process a query and return relevant responses.
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.generate_embedding(query)
        
        # Search vector store
        results = vector_store.search(query_embedding, top_k=top_k)
        
        # Generate response using LLM
        response = llm_model.generate_response(query, results)
        
        return {
            "status": "success",
            "response": response,
            "sources": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 