from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from pathlib import Path
from pydantic import BaseModel

from src.models.document_processor import DocumentProcessor
from src.models.embedding_model import EmbeddingModel
from src.models.llm_model import LLMModel
from src.utils.vector_store import VectorStore
from src.utils.azure_language_service import AzureLanguageService

app = FastAPI(title="Edge RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components
document_processor = DocumentProcessor()
embedding_model = EmbeddingModel()
azure_language_service = AzureLanguageService()
vector_store = VectorStore()

# Pydantic model for query request
class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        chunks = document_processor.process_document(str(file_path))
        
        # Generate embeddings and store in vector database
        for chunk in chunks:
            embedding = embedding_model.generate_embedding(chunk.text)
            vector_store.add_document(
                text=chunk.text,
                embedding=embedding,
                metadata={"source": file.filename}
            )
        
        return {"message": "Document processed and indexed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        # Detect query language
        detected_language = azure_language_service.detect_language(request.query)
        print(f"Detected language for query: {detected_language}")

        # Select LLM model based on detected language
        if detected_language == "ar":
            llm_model_name = "phi4-mini:latest"
        else:
            llm_model_name = "gemma3:1b"
        
        # Initialize LLMModel with the selected model name
        llm_model = LLMModel(model_name=llm_model_name)

        # Generate query embedding
        query_embedding = embedding_model.generate_embedding(request.query)
        
        # Search for relevant documents
        results = vector_store.search(query_embedding, limit=5)
        
        # Generate response using LLM
        context = "\n".join([doc.text for doc in results])
        response = llm_model.generate_response(request.query, context)
        
        return {
            "response": response,
            "sources": [
                {
                    "text": doc.text,
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": doc.score
                }
                for doc in results
            ],
            "detected_language": detected_language,
            "llm_model_used": llm_model_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 