"""
Streamlit Frontend for Edge RAG Application

Provides a simple web interface for:
- Document upload
- Query input
- Response display
"""

import streamlit as st
import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def main():
    st.set_page_config(layout="wide") # Use wide layout

    # Inject CSS for font handling
    st.markdown("""
        <style>
            body {
                font-family: 'Noto Sans Arabic', sans-serif;
            }
            .stExpander, .stTextInput, .stButton, .stText, .stMarkdown, .stSubheader {
                font-family: 'Noto Sans Arabic', sans-serif !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Edge RAG App Powered by Azure AI Containers")
    st.markdown("A lightweight RAG system that provides accurate answers by searching through your documents. No model retraining needed - just upload your files and start asking questions.")

    # # Sidebar for Tech Stack Info
    # st.sidebar.markdown("---\n### Tech Stack")
    # st.sidebar.markdown("📱 **Frontend**: Streamlit\n🚀 **Backend API**: FastAPI\n📄 **Document Processing**: Azure Document Intelligence, PyPDF2 (for PDFs), Python's built-in file handling (for TXT)\n🔍 **Embedding Model**: Ollama (bge-m3)\n🤖 **LLM**: Ollama (gemma3:1b)\n📊 **Vector Database**: Qdrant\n🔧 **Dependency Management**: Python venv, requirements.txt")

    # File upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        # Save the file temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Upload to API
        with open(file_path, "rb") as f:
            files = {"file": (uploaded_file.name, f)}
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                st.success("Document uploaded and processed successfully!")
            else:
                st.error(f"Error uploading document: {response.text}")
    
    # Query interface
    st.header("Ask Questions")
    
    # Initialize session state for the question input
    if "question" not in st.session_state:
        st.session_state["question"] = ""

    # The text input always reads its value from session state
    query = st.text_input("Enter your question:", value=st.session_state["question"], key="main_query_input")
    
    # If the user types directly, update the session state variable
    if "main_query_input" in st.session_state and st.session_state["main_query_input"] != st.session_state["question"]:
        st.session_state["question"] = st.session_state["main_query_input"]
    
    # Ensure the 'query' variable used downstream is always from session state
    query = st.session_state["question"]

    if query:
        # Send query to API
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display response
            st.subheader("Response")
            st.write(result["response"])

            # Display detected language
            if "detected_language" in result:
                st.info(f"Detected Language: {result['detected_language'].upper()}")
            
            # Display LLM model used
            if "llm_model_used" in result:
                st.info(f"LLM Model Used: {result['llm_model_used']}")
            
            # Display sources with more detail
            st.subheader("Sources")
            for i, source_data in enumerate(result["sources"]):
                with st.expander(f"Source {i+1}: {source_data['source']} (Accuracy: {source_data['score']:.2f})"):
                    st.write(source_data['text'])
        else:
            st.error(f"Error getting response: {response.text}")

    # Example Prompts at the bottom
    st.markdown("---\n### Example Prompts")

    example_prompts = [
        {"display": "What is G42's role in the UAE's technological innovation?", "prompt": "What is G42's role in the UAE's technological innovation?"},
        {"display": "صف تعاون مايكروسوفت مع المجموعة 42 في الإمارات.", "prompt": "صف تعاون مايكروسوفت مع المجموعة 42 في الإمارات."}
    ]

    for i, item in enumerate(example_prompts):
        if st.button(item["display"], key=f"prompt_button_{i}"):
            st.session_state["question"] = item["prompt"]
            st.success("Prompt copied to input field!")
            st.rerun() # Rerun to update the input field immediately

if __name__ == "__main__":
    main() 