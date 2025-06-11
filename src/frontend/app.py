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
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000"

def get_indexed_files():
    try:
        response = requests.get(f"{API_URL}/indexed-files")
        response.raise_for_status()
        return response.json()["files"]
    except Exception as e:
        st.error(f"Error fetching indexed files: {e}")
        return []

def main():
    st.set_page_config(layout="wide") # Use wide layout

    # Inject CSS for font handling and styling
    st.markdown("""
        <style>
            body {
                font-family: 'Noto Sans Arabic', sans-serif;
            }
            .stExpander, .stTextInput, .stButton, .stText, .stMarkdown, .stSubheader {
                font-family: 'Noto Sans Arabic', sans-serif !important;
            }
            .file-info {
                font-size: 0.8em;
                color: #666;
                margin-top: 0.2em;
            }
            .stats-box {
                background-color: #f0f2f6;
                padding: 1em;
                border-radius: 0.5em;
                margin: 0.5em 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for indexed files and stats
    with st.sidebar:
        st.header("📚 Indexed Files")
        indexed_files = get_indexed_files()
        
        if indexed_files:
            # Show file count
            st.markdown(f"**Total Files:** {len(indexed_files)}")
            
            # Show file types
            file_types = {}
            for file in indexed_files:
                ext = Path(file).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            st.markdown("**File Types:**")
            for ext, count in file_types.items():
                st.markdown(f"- {ext}: {count} files")
            
            # List files with icons based on type
            st.markdown("---")
            st.markdown("**Files:**")
            for file in indexed_files:
                ext = Path(file).suffix.lower()
                icon = "📄"  # default
                if ext == ".pdf":
                    icon = "📑"
                elif ext == ".txt":
                    icon = "📝"
                elif ext == ".doc" or ext == ".docx":
                    icon = "📘"
                
                st.markdown(f"{icon} {file}")
                # Add file size and last modified if available
                try:
                    file_path = Path("uploads") / file
                    if file_path.exists():
                        size = file_path.stat().st_size / 1024  # size in KB
                        modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                        st.markdown(f"<div class='file-info'>Size: {size:.1f}KB | Modified: {modified.strftime('%Y-%m-%d %H:%M')}</div>", 
                                  unsafe_allow_html=True)
                except:
                    pass
        else:
            st.info("No files indexed yet. Upload a document to get started!")

    st.title("Edge RAG App Powered by Azure AI Containers")
    st.markdown("A lightweight RAG system that provides accurate answers by searching through your documents. No model retraining needed - just upload your files and start asking questions.")

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
                # Refresh the page to update the sidebar
                st.rerun()
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
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            )
            response.raise_for_status()
            result = response.json()
            
            # Display response
            st.subheader("Response")
            if result["detected_language"] == "ar":
                st.markdown(f"<div style=\"direction: rtl; text-align: right;\">{result["response"]}</div>", unsafe_allow_html=True)
            else:
                st.write(result["response"])

            # Display detected language and model info in a stats box
            with st.container():
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                if "detected_language" in result:
                    st.info(f"🌐 Detected Language: {result['detected_language'].upper()}")
                if "llm_model_used" in result:
                    st.info(f"🤖 LLM Model Used: {result['llm_model_used']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources with more detail
            st.subheader("Sources")
            for i, source_data in enumerate(result["sources"]):
                with st.expander(f"Source {i+1}: {source_data['source']} (Relevance: {source_data['score']:.2f})"):
                    st.write(source_data['text'])
                    # Add source metadata if available
                    if "metadata" in source_data:
                        st.markdown("**Metadata:**")
                        for key, value in source_data["metadata"].items():
                            st.markdown(f"- {key}: {value}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please ensure the backend is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Error from backend: {e.response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

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
            st.rerun()

if __name__ == "__main__":
    main() 