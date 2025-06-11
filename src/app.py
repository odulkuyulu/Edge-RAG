"""
Streamlit Frontend for Edge RAG Application.

This module provides the user interface for interacting with the Edge RAG solution.
It handles document uploads, allows users to ask questions, and displays responses
along with relevant sources and system statistics.
"""

import streamlit as st
import requests
import json
from pathlib import Path
from datetime import datetime

# Define the backend API endpoint URL
API_URL = "http://localhost:8000"

def get_indexed_files() -> list:
    """
    Fetches the list of currently indexed files from the backend API.

    Returns:
        list: A list of filenames that are indexed in the vector database.
    """
    try:
        # Make a GET request to the backend API to retrieve indexed files
        response = requests.get(f"{API_URL}/indexed-files")
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()["files"]
    except Exception as e:
        # Display an error message if fetching files fails
        st.error(f"Error fetching indexed files: {e}")
        return []

def main():
    """
    Main function to run the Streamlit application.
    Configures the page, handles UI elements, and manages interactions with the backend API.
    """
    # Set Streamlit page configuration to use a wide layout
    st.set_page_config(layout="wide")

    # Inject custom CSS for styling and font handling
    # This ensures proper display of Arabic text (right-to-left alignment, specific font)
    st.markdown("""
        <style>
            /* Ensure Noto Sans Arabic is used for general body text */
            body {
                font-family: 'Noto Sans Arabic', sans-serif;
            }
            /* Apply Noto Sans Arabic to specific Streamlit components */
            .stExpander, .stTextInput, .stButton, .stText, .stMarkdown, .stSubheader {
                font-family: 'Noto Sans Arabic', sans-serif !important;
            }
            /* Styling for file information in the sidebar */
            .file-info {
                font-size: 0.8em;
                color: #666;
                margin-top: 0.2em;
            }
            /* Styling for the statistics display box */
            .stats-box {
                background-color: #f0f2f6;
                padding: 1em;
                border-radius: 0.5em;
                margin: 0.5em 0;
            }
        </style>
    """, unsafe_allow_html=True) # unsafe_allow_html is required to inject custom HTML/CSS

    # Sidebar for displaying indexed files and summary statistics
    with st.sidebar:
        st.header("📚 Indexed Files")
        indexed_files = get_indexed_files()
        
        if indexed_files:
            # Display the total count of indexed files
            st.markdown(f"**Total Files:** {len(indexed_files)}")
            
            # Categorize and display file types
            file_types = {}
            for file in indexed_files:
                ext = Path(file).suffix.lower() # Extract file extension
                file_types[ext] = file_types.get(ext, 0) + 1
            
            st.markdown("**File Types:**")
            for ext, count in file_types.items():
                st.markdown(f"- {ext}: {count} files")
            
            st.markdown("---") # Separator
            st.markdown("**Files:**")
            
            # List each indexed file with an appropriate icon and optional metadata
            for file in indexed_files:
                ext = Path(file).suffix.lower() # Get file extension for icon selection
                icon = "📄"  # Default icon
                if ext == ".pdf":
                    icon = "📑"
                elif ext == ".txt":
                    icon = "📝"
                elif ext == ".doc" or ext == ".docx":
                    icon = "📘"
                
                st.markdown(f"{icon} {file}") # Display icon and filename
                
                # Attempt to retrieve and display file size and last modified date
                try:
                    file_path = Path("uploads") / file # Assuming uploaded files are in the 'uploads' directory
                    if file_path.exists():
                        size = file_path.stat().st_size / 1024  # Size in KB
                        modified = datetime.fromtimestamp(file_path.stat().st_mtime) # Last modified timestamp
                        st.markdown(f"<div class='file-info'>Size: {size:.1f}KB | Modified: {modified.strftime('%Y-%m-%d %H:%M')}</div>", 
                                  unsafe_allow_html=True) # Use HTML for custom styling
                except Exception:
                    # Silently pass if file info cannot be retrieved (e.g., file moved or deleted)
                    pass
        else:
            st.info("No files indexed yet. Upload a document to get started!")

    # Main content area - Application Title and Description
    st.title("Edge RAG App Powered by Azure AI Containers")
    st.markdown("A lightweight RAG system that provides accurate answers by searching through your documents. No model retraining needed - just upload your files and start asking questions.")

    # Section for file upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"]) # Added docx support for UI
    
    if uploaded_file is not None:
        # Create a temporary directory to store the uploaded file before sending to API
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True) # Create if not exists
        file_path = temp_dir / uploaded_file.name
        
        # Write the uploaded file content to the temporary path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Send the file to the backend API for processing and indexing
        with open(file_path, "rb") as f:
            files = {"file": (uploaded_file.name, f)} # Prepare file for multipart/form-data upload
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                st.success("Document uploaded and processed successfully!")
                st.rerun() # Rerun the app to refresh the sidebar with the new file
            else:
                st.error(f"Error uploading document: {response.text}")

    # Section for query interface
    st.header("Ask Questions")
    
    # Initialize session state variable for the question input if it doesn't exist
    # This helps retain the input value across reruns
    if "question" not in st.session_state:
        st.session_state["question"] = ""

    # Text input for user query. It's tied to session state for persistence.
    query = st.text_input("Enter your question:", value=st.session_state["question"], key="main_query_input")
    
    # Update session state if the user directly modifies the text input
    if "main_query_input" in st.session_state and st.session_state["main_query_input"] != st.session_state["question"]:
        st.session_state["question"] = st.session_state["main_query_input"]
    
    # Ensure the 'query' variable always reflects the latest session state value
    query = st.session_state["question"]

    # Process query if input is not empty
    if query:
        try:
            # Send the user query to the backend API
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query} # Send query as JSON payload
            )
            response.raise_for_status() # Check for HTTP errors
            result = response.json() # Parse the JSON response
            
            # Display the LLM's response
            st.subheader("Response")
            if result["detected_language"] == "ar":
                # Apply RTL (Right-to-Left) and right alignment for Arabic text display
                st.markdown(f'<div style="direction: rtl; text-align: right;">{result["response"]}</div>', unsafe_allow_html=True)
            else:
                st.write(result["response"])

            # Display detected language and LLM model used in a styled box
            with st.container(): # Use a container for grouped styling
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                if "detected_language" in result:
                    st.info(f"🌐 Detected Language: {result['detected_language'].upper()}")
                if "llm_model_used" in result:
                    st.info(f"🤖 LLM Model Used: {result['llm_model_used']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources of information in expandable sections
            st.subheader("Sources")
            for i, source_data in enumerate(result["sources"]):
                # Create an expander for each source, showing filename and relevance score
                with st.expander(f"Source {i+1}: {source_data['source']} (Relevance: {source_data['score']:.2f})"):
                    st.write(source_data['text']) # Display the text content of the source
                    
                    # Display additional metadata if available
                    if "metadata" in source_data:
                        st.markdown("**Metadata:**")
                        for key, value in source_data["metadata"].items():
                            st.markdown(f"- {key}: {value}")
        
        except requests.exceptions.ConnectionError:
            # Handle cases where the backend API is not reachable
            st.error("Could not connect to the backend API. Please ensure the backend is running.")
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors returned by the backend API
            st.error(f"Error from backend: {e.response.text}")
        except Exception as e:
            # Catch any other unexpected errors
            st.error(f"An unexpected error occurred: {e}")

    # Section for example prompts to guide the user
    st.markdown("---\n### Example Prompts")

    # Define a list of example prompts with display text and actual prompt text
    example_prompts = [
        {"display": "What is G42's role in the UAE's technological innovation?", "prompt": "What is G42's role in the UAE's technological innovation?"},
        {"display": "صف تعاون مايكروسوفت مع المجموعة 42 في الإمارات.", "prompt": "صف تعاون مايكروسوفت مع المجموعة 42 في الإمارات"}
    ]

    # Create a button for each example prompt
    for i, item in enumerate(example_prompts):
        if st.button(item["display"], key=f"prompt_button_{i}"):
            st.session_state["question"] = item["prompt"] # Set the text input value via session state
            st.success("Prompt copied to input field!")
            st.rerun() # Rerun the app to update the text input field

# Entry point for the Streamlit application
if __name__ == "__main__":
    main() 