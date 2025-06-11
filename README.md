# Edge RAG Solution: Local-First & Deployable Anywhere

This project provides a robust **Retrieval-Augmented Generation (RAG)** application designed for **edge environments** and **offline capabilities**. It leverages local LLMs (Ollama) and vector databases (Qdrant), alongside enterprise-grade Azure AI services (disconnected containers) for advanced document processing. This solution enables secure, private, and efficient querying of your documents without continuous internet connectivity.

## Key Features

*   **Local-First & Offline**: All core components run on your device, enabling complete offline operation.
*   **Flexible Deployment**: Portable and designed for various environments.
*   **Advanced Document Processing**: Utilizes Azure Document Intelligence (disconnected containers) for high-accuracy text and structure extraction from PDFs.
*   **Multi-language Support**: Detects query language and uses optimized LLMs (e.g., Arabic).
*   **Local LLM & Vector DB**: Powered by Ollama for LLM inference and Qdrant for vector search.
*   **Intuitive UI**: Streamlit web interface for document upload and querying.

## Quick Start

To get this RAG solution up and running:

1.  **Prerequisites**: Ensure you have Python 3.9+, Docker, and Ollama installed.

2.  **Clone & Setup**:  
    ```bash
    git clone https://github.com/yourusername/edge-rag.git
    cd edge-rag
    python3 -m venv venv
    source venv/bin/activate # macOS/Linux
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
    # Start Qdrant via Docker Compose (if not already running):
    docker compose up -d
    ```

5.  **Clear Qdrant (Optional, for fresh start)**:
    ```bash
    venv/bin/python src/utils/clear_qdrant.py
    ```
    
6.  **Start Application**:  
    ```bash
    # In one terminal for backend API
    venv/bin/uvicorn src.api.main:app --reload

    # In another terminal for Streamlit frontend
    venv/bin/streamlit run src/frontend/app.py
    ```
    Access the app at `http://localhost:8501`.

## Project Structure

```
src/
├── api/
│   └── main.py              # FastAPI backend
├── frontend/
│   └── app.py               # Streamlit web app
├── models/
│   ├── document_processor.py # Document processing (Azure DI)
│   ├── embedding_model.py   # Embedding generation (Ollama)
│   └── llm_model.py         # LLM response generation (Ollama)
└── utils/
    ├── clear_qdrant.py      # Clear Qdrant database
    ├── azure_language_service.py # Azure Language Detection
    └── vector_store.py      # Qdrant interaction
```

## Technology Stack

*   **Frontend**: Streamlit
*   **Backend API**: FastAPI
*   **Document Processing**: Azure AI Document Intelligence Service
*   **Language Detection**: Azure AI Language Service
*   **Embedding Model**: Ollama (`bge-m3`)
*   **Large Language Model (LLM)**: Ollama (`gemma3:1b`, `phi4-mini:latest`)
*   **Vector Database**: Qdrant