# Edge RAG Application

A local RAG (Retrieval-Augmented Generation) application that runs entirely on your machine, using Ollama for embeddings and LLM, and Qdrant for vector storage.

## Features

- Document upload and processing
- Text chunking and embedding generation
- Vector similarity search
- Question answering using local LLM
- Simple web interface with Streamlit

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Qdrant running locally

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/edge-rag.git
cd edge-rag
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows

3. Install dependencies:
```bash
pip install -r requirements.txt
source .env 

```

4. Start Ollama and pull required models:
```bash
ollama pull bge-m3
ollama pull gemma3:1b
```

5. Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Running the Application

1. Start the FastAPI backend (in one terminal):
```bash
venv/bin/uvicorn src.api.main:app --reload
```

2. Start the Streamlit frontend (in a separate terminal):
```bash
venv/bin/streamlit run src/frontend/app.py
```

3. Open your browser and navigate to http://localhost:8501

## Usage

1. Upload a document (PDF or text file)
2. Wait for the document to be processed
3. Ask questions about the document's content
4. View the response and source documents

## Project Structure

```
src/
├── api/
│   └── main.py           # FastAPI backend
├── frontend/
│   └── app.py           # Streamlit frontend
├── models/
│   ├── document_processor.py
│   ├── embedding_model.py
│   └── llm_model.py
└── utils/
    └── vector_store.py
```