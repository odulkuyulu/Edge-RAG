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
            
            # Display sources
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(f"- {source}")
        else:
            st.error(f"Error getting response: {response.text}")

if __name__ == "__main__":
    main() 