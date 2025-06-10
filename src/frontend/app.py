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
    st.title("Edge RAG Application")

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
    query = st.text_input("Enter your question:")
    
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
            
            # Display sources with more detail
            st.subheader("Sources")
            for i, source_data in enumerate(result["sources"]):
                with st.expander(f"Source {i+1}: {source_data['source']} (Accuracy: {source_data['score']:.2f})"):
                    st.write(source_data['text'])
        else:
            st.error(f"Error getting response: {response.text}")

    # Example Prompts at the bottom
    st.markdown("---\n### Example Prompts")
    st.markdown("**English:** What is G42's role in the UAE's technological innovation?")
    st.markdown("**Arabic:** ما هو التعاون بين G42 ومايكروسوفت؟")

if __name__ == "__main__":
    main() 