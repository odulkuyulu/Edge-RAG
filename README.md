# Edge RAG Solution: Local-First & Deployable Anywhere

This project provides a robust **Retrieval-Augmented Generation (RAG)** application designed for **edge environments** and **offline capabilities**. Leveraging powerful open-source tools and enterprise-grade Azure AI services (including disconnected containers), it allows you to deploy a fully functional RAG solution that can process your private documents and answer queries without continuous internet connectivity, making it ideal for sensitive data, remote locations, or environments with strict network policies.

## Features

-   **Local-First Architecture:** All core components run on your local machine or edge device.
-   **Offline Operation:** Perform document processing, embedding generation, vector search, and LLM inference entirely offline.
-   **Flexible Deployment:** Designed for portability, enabling deployment in various environments, from on-premises servers to specialized edge hardware.
-   **Enterprise-Grade Document Processing (with Azure AI Disconnected Containers):**
    -   Leverages Azure Document Intelligence for highly accurate text, layout, and structure extraction from diverse PDF documents, even in disconnected environments.
    -   Supports various file types including PDFs and plain text files.
-   **Local LLM Inference (Ollama):** Utilizes Ollama for efficient and private large language model operations.
-   **Efficient Vector Storage (Qdrant):** Manages document embeddings for fast and scalable semantic search.
-   **Intuitive Web Interface (Streamlit):** A user-friendly UI for document upload and querying.
-   **Dynamic Source Attribution:** Provides accurate sources and similarity scores for generated responses.

## Prerequisites

-   **Python 3.8+**
-   **Docker** (for Qdrant and Azure AI Disconnected Containers)
-   **Ollama** installed and running locally
-   **Azure Account & Permissions (for Azure AI Disconnected Containers):** Access to Azure to provision and download Azure AI Document Intelligence disconnected containers. This is crucial for offline PDF processing.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/edge-rag.git
    cd edge-rag
    ```

2.  **Create and activate a Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # macOS/Linux
    venv\Scripts\activate # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables (`.env` file):**
    Create a `.env` file in the root directory of your project. This file will store sensitive credentials and configuration.

    ```dotenv
    # Azure AI Language Service (Optional - for future integration if needed)
    AZURE_LANGUAGE_ENDPOINT=
    AZURE_LANGUAGE_KEY=

    # Azure Document Intelligence (REQUIRED for PDF processing with disconnected containers)
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
    AZURE_DOCUMENT_INTELLIGENCE_KEY=

    # Qdrant Configuration
    QDRANT_URL=http://localhost:6333

    # Processing Configuration
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    EMBEDDING_SIZE=1024
    SEARCH_LIMIT=5
    SCORE_THRESHOLD=0.7
    ```

5.  **Set up Azure AI Document Intelligence Disconnected Container:**
    *   **Acquire the Container:** Follow the official Microsoft Azure documentation to acquire and deploy the Azure Document Intelligence disconnected container. This typically involves requesting access, downloading the container image, and running it locally via Docker.
    *   **Configure `.env`:** Ensure the `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENT_INTELLIGENCE_KEY` in your `.env` file point to your running disconnected container.

6.  **Ensure Ollama is Running:**
    Make sure you have Ollama installed and the `bge-m3` (embedding) and `gemma3:1b` (LLM) models pulled. If not, run:
    ```bash
    ollama pull bge-m3
    ollama pull gemma3:1b
    ```
    Ensure the Ollama server is running in the background (e.g., `ollama serve`).

7.  **Ensure Qdrant is Running:**
    Qdrant is set up to run via Docker Compose. Navigate to the root of the project and run:
    ```bash
    docker compose up -d
    ```
    To verify Qdrant is running, you can use `docker ps`.

8.  **Clear Qdrant Database (Recommended for Fresh Start):**
    Before indexing new documents, especially for demos or fresh tests, it's recommended to clear any existing data in Qdrant:
    ```bash
    venv/bin/python src/utils/clear_qdrant.py
    ```

## Running the Application

1.  **Start the FastAPI Backend:**
    In a new terminal, navigate to the project root and run:
    ```bash
    venv/bin/uvicorn src.api.main:app --reload
    ```
    This will start the backend API, typically accessible at `http://127.0.0.1:8000`. This process will use Azure Document Intelligence (if configured) for PDF processing and Ollama for embeddings and LLM inference.

2.  **Start the Streamlit Frontend:**
    In a separate terminal, navigate to the project root and run:
    ```bash
    venv/bin/streamlit run src/frontend/app.py
    ```
    This will open the Streamlit web application in your browser, typically at `http://localhost:8501`.

3.  **Usage:**
    *   Upload your documents (PDF or TXT) via the Streamlit UI.
    *   Ask questions related to their content. The system will retrieve relevant information and generate responses, even in an offline environment (after initial setup).

## Project Structure

```
src/
├── api/
│   └── main.py           # FastAPI backend application, defines API endpoints
├── frontend/
│   └── app.py           # Streamlit web application for user interaction
├── models/
│   ├── document_processor.py # Handles document text extraction and chunking (integrates Azure DI)
│   ├── embedding_model.py # Manages text embedding generation via Ollama
│   └── llm_model.py # Manages LLM response generation via Ollama
└── utils/
    ├── clear_qdrant.py  # Utility script to clear the Qdrant vector database
    └── vector_store.py # Manages interaction with the Qdrant vector database
```

## Technology Stack & Methods

This Edge RAG application is built with a local-first approach, leveraging powerful open-source tools and enterprise-grade Azure AI services to provide a complete Retrieval-Augmented Generation system that is deployable anywhere and supports offline operations.

### Key Technologies

*   **Frontend:**
    *   **Streamlit**: Provides a simple and interactive web interface for users to upload documents and ask questions.

*   **Backend API:**
    *   **FastAPI**: A modern, high-performance web framework for building the robust API endpoints that handle document uploads, processing, and query responses.

*   **Document Processing:**
    *   **Azure AI Document Intelligence (Disconnected Containers)**: The cornerstone for robust PDF processing in disconnected environments. It enables highly accurate text, layout, and structure extraction on-premises, critical for maintaining data sovereignty and supporting offline scenarios.
    *   **PyPDF2**: Utilized for basic text extraction from PDF documents when Azure Document Intelligence is not configured or for fallback (though the current setup prioritizes Azure DI).
    *   **Python's built-in file handling**: Used for processing plain text (`.txt`) files.

*   **Embedding Model:**
    *   **Ollama (`bge-m3` model)**: A local large language model runner that hosts the `bge-m3` model. This model converts text (document chunks and user queries) into dense numerical vectors (embeddings), enabling semantic similarity search entirely offline.

*   **Large Language Model (LLM):**
    *   **Ollama (`gemma3:1b` model)**: Also hosted by Ollama, the `gemma3:1b` model is used to generate coherent and contextually relevant text responses based on the retrieved information and the user's query, operating completely offline.

*   **Vector Database:**
    *   **Qdrant**: An open-source vector similarity search engine that stores the generated vector embeddings locally. It's optimized for high-performance similarity searches, allowing for fast retrieval of relevant document chunks without external dependencies.

*   **Dependency Management & Environment:**
    *   **Python `venv`**: Used to create isolated Python environments, ensuring project dependencies don't conflict with system-wide packages.
    *   **`requirements.txt`**: Lists all Python libraries and their specific versions required for the project.

### Core Methods & Concepts

*   **Retrieval-Augmented Generation (RAG)**: The overarching architecture. Instead of the LLM generating responses solely from its training data, it first retrieves relevant information from your private documents (stored locally), and then uses that information to augment its generation process, leading to more accurate and up-to-date answers in any environment.
*   **Document Chunking**: Large documents are split into smaller, manageable text blocks (chunks) to optimize retrieval. This helps in pinpointing precise information relevant to a query.
*   **Text Embedding**: The process of converting text (both document chunks and user queries) into numerical vectors (embeddings) using an embedding model. These vectors capture the semantic meaning, allowing for mathematical comparison of similarity.
*   **Vector Search**: When a user submits a query, its embedding is used to search the local vector database (Qdrant) for the most semantically similar document chunks.
*   **LLM Inference**: The process where the retrieved relevant document chunks are provided as context to the local LLM (Gemma), along with the user's original query, to generate a comprehensive and informed response.