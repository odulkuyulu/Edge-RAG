# AI Edge RAG Solution

This project provides a robust **Retrieval-Augmented Generation (RAG)** application designed for **edge environments** and **offline capabilities**. It leverages local LLMs (Ollama) and vector databases (Qdrant), alongside enterprise-grade Azure AI services (disconnected containers) for advanced document processing. This solution enables secure, private, and efficient querying of your documents without continuous internet connectivity.

## Project Structure

```
src/
├── api.py                   # FastAPI backend API
├── app.py                   # Streamlit web application
├── embeddings.py            # Text embedding model (Ollama)
├── indexer.py               # Document indexing logic (Azure DI, Ollama embeddings, Qdrant storage)
├── retriever.py             # RAG query and LLM response generation (Azure Language, Ollama LLM, Qdrant search)
├── vector_db.py             # Qdrant vector database client
└── vector_db_cleaner.py     # Script to clear Qdrant database
```

## Key Features

*   **Local-First & Offline**: All core components run on your device, enabling complete offline operation.
*   **Flexible Deployment**: Portable and designed for various environments.
*   **Advanced Document Processing**: Utilizes Azure Document Intelligence (disconnected containers) for high-accuracy text and structure extraction from PDFs.
*   **Multi-language Support**: Detects query language and uses optimized LLMs (e.g., Arabic).
*   **Local LLM & Vector DB**: Powered by Ollama for LLM inference and Qdrant for vector search.
*   **Intuitive UI**: Streamlit web interface for document upload and querying.

## Getting Started

1.  **Prerequisites**: Ensure you have Python 3.9+, Docker, and Ollama installed.

2.  **Clone & Setup**:  
    ```bash
    git clone https://github.com/hamza-roujdami/edge-rag.git
    cd edge-rag
    python3 -m venv venv
    source venv/bin/activate 
    pip install -r requirements.txt
    ```

3.  **Configure `.env`**: Set up `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, `AZURE_DOCUMENT_INTELLIGENCE_KEY`, `AZURE_LANGUAGE_ENDPOINT`, `AZURE_LANGUAGE_KEY`, and `QDRANT_URL` in a `.env` file at the project root.

4.  **Run Dependencies**:  
    ```bash
    # Ensure Ollama server is running and models are pulled:
    ollama serve
    ollama pull bge-m3
    ollama pull gemma3:1b
    ollama pull phi4-mini:latest # For Arabic queries
    # Start Qdrant 
    docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
    ```

5.  **Clear Qdrant (Optional, for fresh start)**:
    ```bash
    venv/bin/python src/vector_db_cleaner.py
    ```
    
6.  **Start Application**:  
    ```bash
    # In one terminal for backend API
    uvicorn src.api:app --reload

    # In another terminal for Streamlit frontend
    streamlit run src/app.py
    ```
    Access the app at `http://localhost:8501`.

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive web interface for document upload and querying |
| Backend API | FastAPI | RESTful API for document processing and query handling |
| Document Processing | Azure AI Document Intelligence | High-accuracy text extraction from PDFs and documents |
| Language Detection | Azure AI Language Service | Automatic language detection for multi-language support |
| Embedding Model | Ollama (bge-m3) | Text embedding generation for semantic search |
| LLM Models | Ollama (gemma3:1b, phi4-mini) | Language model for response generation (English and Arabic) |
| Vector Database | Qdrant | Efficient vector storage and similarity search |