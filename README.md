# Offline RAG: Local AI Assistant

A powerful **Retrieval-Augmented Generation (RAG)** system that brings enterprise-grade AI capabilities to your local environment. Built with privacy and offline capabilities as core principles, this solution combines the power of local LLMs (Ollama) with Azure AI services in disconnected containers.

## Why Offline RAG?

- 🔒 **Privacy-First**: Process sensitive documents locally without sending data to the cloud
- 💻 **Fully Offline**: Complete functionality without any internet connectivity
- 🚀 **Enterprise-Grade**: Leverages Azure AI services in disconnected containers for advanced document processing
- 🌍 **Multi-Language**: Native support for multiple languages with optimized models
- 🛠️ **Developer-Friendly**: Simple setup and intuitive API

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

For a facilitator-led, step-by-step workshop, use the
[Partner Enablement Guide](./PARTNER_ENABLEMENT_GUIDE.md).

### Azure AI containers on an x64 Windows Docker host

Use `compose.local-ai.yaml` on a Linux x64 Docker engine (for example Docker
Desktop on an x64 Windows VM). The stack includes Language detection, NER,
key phrases, English sentiment, Document Intelligence General Document v3.0
with its Layout dependency, a separate persistent Qdrant database, and Ollama.
Allow approximately 58 GiB for the configured container memory limits,
plus room for the host application and Ollama.

Keep the Azure cloud billing endpoints and keys in the ignored `.env`.
Do not replace them with localhost: the containers use those values for
metering. These are connected, metered containers, **not licensed
disconnected containers**. Starting them accepts Microsoft's container EULA.

```powershell
docker compose -f .\compose.local-ai.yaml up -d
docker compose -f .\compose.local-ai.yaml ps
```

Install the enhanced-demo Python dependencies described below. The Ollama
container mounts the host's existing `%USERPROFILE%\.ollama\models`
read-only; download `bge-m3` and `qwen2.5:0.5b` with a working host Ollama
installation first, or set `OLLAMA_MODELS_PATH` to a populated model directory.
It exposes port 11435 without replacing a host Ollama installation on 11434.
Then, from the repository directory, run:

```powershell
.\Start-LocalApp.ps1 -PythonPath .\venv\Scripts\python.exe -IndexOnly
.\Start-LocalApp.ps1 -PythonPath .\venv\Scripts\python.exe
```

The launcher overrides application endpoints only for its child process.
`AZURE_AI_CONTAINER_MODE=true` routes both indexing and retrieval to
feature-specific local Language endpoints on ports 5000-5003. PDF analysis
uses port 5010 and API version `2022-08-31`, matching the General Document
v3.0 container. All published ports bind to loopback; open
http://localhost:8503 in the Docker host's browser, including through a
Bastion desktop session when using a VM.

The default cloud behavior is unchanged when container mode is not enabled.
Local Language errors and PDF container failures are reported rather than
silently falling back to cloud inference or another PDF extractor.
Language-model support differs between container images; this initial stack
uses the English NER and sentiment images. Arabic enrichment requires
compatible language-specific containers and validation; do not assume
cloud multilingual parity. The local UI displays this limitation and the
launcher's indexing command explicitly skips non-English documents.
`tests/verify_local_services.py` exercises every local inference service,
including real PDF analysis, before indexing.
After indexing the bundled English samples into a fresh volume, run
`python .\tests\verify_local_services.py --rag` to verify the expected eight
chunks and a real Streamlit search-and-generation flow.

For troubleshooting, use `docker compose -f .\compose.local-ai.yaml logs
--tail 50 SERVICE`. Avoid printing `docker compose config` or container
environment variables: they contain billing credentials. To stop only this
stack, use `docker compose -f .\compose.local-ai.yaml down`; indexed data
remains in its named volume.

### Windows PowerShell: enhanced demo

Run these commands from the repository root. This configuration is a
**hybrid/cloud demo**, not disconnected operation: the Azure endpoints in
`.env` receive document and query content. Both Azure resources must permit
key-based authentication for the current application.

Use Python 3.12 x64 on Windows, including Windows ARM devices where native
Python ARM dependency wheels may be unavailable. Create a virtual environment
with that interpreter, then install the dependencies used by the enhanced app:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install ollama "qdrant-client<1.16" streamlit rank-bm25 nltk azure-ai-formrecognizer "azure-core==1.30.1" "azure-ai-textanalytics==5.3.0" PyPDF2 python-dotenv requests numpy
```

The broader dependency list includes packages not used by this demo. The
Qdrant client must remain below 1.16 because the retrievers use its removed
legacy `search()` method.

Start Qdrant in Docker Desktop's Linux-container mode:

```powershell
docker run --name edge-rag-qdrant -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 -v edge-rag-qdrant-storage:/qdrant/storage qdrant/qdrant:v1.15.5
```

For subsequent runs, use `docker start -a edge-rag-qdrant`. Keep Ollama
running, either through its desktop app or `ollama serve` in another terminal.
Download the default enhanced-app models:

```powershell
ollama pull bge-m3
ollama pull qwen2.5:0.5b
```

With Qdrant and Ollama ready, index the files in `data` once:

```powershell
.\venv\Scripts\python.exe .\src\enhanced_indexer.py
```

Confirm that each document has successfully indexed chunks. Re-running
indexing adds new points; it does not replace previously indexed documents.
Then launch the UI:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\src\enhanced_app.py --server.port 8503 --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://localhost:8503. Separate Language detection and NER containers
are not used when `.env` points to Azure cloud endpoints. Using local
containers requires additional routing because the current code shares one
Language endpoint across detection, entities, key phrases, and sentiment.

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
