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
```

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
venv/bin/python src/utils/clear_qdrant.py // clear the qdrant
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
│   └── main.py           # FastAPI backend application, defines API endpoints
├── frontend/
│   └── app.py           # Streamlit web application for user interaction
├── models/
│   ├── document_processor.py # Handles document text extraction and chunking
│   ├── embedding_model.py # Manages text embedding generation via Ollama
│   └── llm_model.py # Manages LLM response generation via Ollama
└── utils/
    └── vector_store.py # Manages interaction with the Qdrant vector database
```

## Technology Stack & Methods

This Edge RAG application is built with a local-first approach, leveraging powerful open-source tools to provide a complete Retrieval-Augmented Generation system.

### Key Technologies

*   **Frontend:**
    *   **Streamlit**: Provides a simple and interactive web interface for users to upload documents and ask questions.

*   **Backend API:**
    *   **FastAPI**: A modern, high-performance web framework for building the robust API endpoints that handle document uploads, processing, and query responses.

*   **Document Processing:**
    *   **PyPDF2**: Utilized for extracting text content from PDF documents.
    *   **Python's built-in file handling**: Used for processing plain text (`.txt`) files.

*   **Embedding Model:**
    *   **Ollama (`bge-m3` model)**: A local large language model runner that hosts the `bge-m3` model. This model converts text (document chunks and user queries) into dense numerical vectors (embeddings), enabling semantic similarity search.

*   **Large Language Model (LLM):**
    *   **Ollama (`gemma3:1b` model)**: Also hosted by Ollama, the `gemma3:1b` model is used to generate coherent and contextually relevant text responses based on the retrieved information and the user's query.

*   **Vector Database:**
    *   **Qdrant**: An open-source vector similarity search engine that stores the generated vector embeddings. It's optimized for high-performance similarity searches, allowing for fast retrieval of relevant document chunks.

*   **Dependency Management & Environment:**
    *   **Python `venv`**: Used to create isolated Python environments, ensuring project dependencies don't conflict with system-wide packages.
    *   **`requirements.txt`**: Lists all Python libraries and their specific versions required for the project.

### Core Methods & Concepts

*   **Retrieval-Augmented Generation (RAG)**: The overarching architecture. Instead of the LLM generating responses solely from its training data, it first retrieves relevant information from your private documents, and then uses that information to augment its generation process, leading to more accurate and up-to-date answers.
*   **Document Chunking**: Large documents are split into smaller, manageable text blocks (chunks) to optimize retrieval. This helps in pinpointing precise information relevant to a query.
*   **Text Embedding**: The process of converting text (both document chunks and user queries) into numerical vectors (embeddings) using an embedding model. These vectors capture the semantic meaning, allowing for mathematical comparison of similarity.
*   **Vector Search**: When a user submits a query, its embedding is used to search the vector database (Qdrant) for the most semantically similar document chunks.
*   **LLM Inference**: The process where the retrieved relevant document chunks are provided as context to the LLM (Gemma), along with the user's original query, to generate a comprehensive and informed response.