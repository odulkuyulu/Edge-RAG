"""
FastAPI Application for Edge RAG Solution.

This module defines the API endpoints for document upload, querying, and managing
indexed files within the Edge RAG (Retrieval-Augmented Generation) system.
It integrates with the Indexer, Retriever, and VectorDBClient components.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from pathlib import Path
from pydantic import BaseModel

from src.indexer import Indexer
from src.retriever import Retriever
from src.vector_db import VectorDBClient

# Initialize the FastAPI application
app = FastAPI(title="Edge RAG API")

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows requests from any origin to access the API, which is useful for development.
# For production, it's recommended to restrict allow_origins to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize core components of the RAG system
indexer = Indexer()
retriever = Retriever()
vector_db_client = VectorDBClient()

# Pydantic model for incoming query requests
class QueryRequest(BaseModel):
    """
    Represents the structure of a query request.

    Attributes:
        query (str): The user's query string.
    """
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document for processing and indexing.

    Args:
        file (UploadFile): The uploaded file to be processed.

    Returns:
        dict: A dictionary containing a message and the number of chunks processed.

    Raises:
        HTTPException: If an error occurs during file processing or indexing.
    """
    try:
        # Ensure the 'uploads' directory exists to store incoming files
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Define the full path for the uploaded file
        file_path = upload_dir / file.filename
        
        # Asynchronously read and write the file content to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process and index the document using the Indexer component
        result = indexer.process_and_index_document(str(file_path), file.filename)
        
        # Return a success message and the number of chunks processed
        return {"message": result.get("message", "Document processed and indexed successfully"), "chunks_processed": result.get("chunks_processed", 0)}
    
    except Exception as e:
        # Catch any exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Processes a user query and returns a generated response with sources.

    Args:
        request (QueryRequest): The query request containing the user's question.

    Returns:
        dict: A dictionary containing the LLM's response, relevant sources, detected language, and LLM model used.

    Raises:
        HTTPException: If an error occurs during the query processing or response generation.
    """
    try:
        # Use the Retriever component to get relevant context and generate an LLM response
        result = retriever.retrieve_and_generate_response(request.query)
        
        # Return the generated response, sources, detected language, and LLM model used
        return {
            "response": result["response"],
            "sources": result["sources"],
            "detected_language": result["detected_language"],
            "llm_model_used": result["llm_model_used"]
        }
    
    except Exception as e:
        # Catch any exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexed-files")
async def get_indexed_files():
    """
    Retrieves a list of all unique source files that have been indexed in the vector database.

    Returns:
        dict: A dictionary containing a list of indexed file names.

    Raises:
        HTTPException: If an error occurs while fetching indexed files.
    """
    try:
        # Use the VectorDBClient to retrieve all unique source file names
        files = vector_db_client.get_unique_sources()
        return {"files": files}
    except Exception as e:
        # Catch any exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))

# This block allows running the FastAPI app directly using 'python src/api.py'
# In a production environment, uvicorn is typically run via command line (e.g., uvicorn src.api:app --host 0.0.0.0 --port 8000)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 