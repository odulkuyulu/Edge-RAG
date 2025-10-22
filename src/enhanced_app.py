"""
Enhanced Streamlit App for Edge RAG System
Leverages intelligent chunking and enhanced retrieval for better accuracy
"""

import streamlit as st
import requests
import os
import json
from datetime import datetime
from enhanced_indexer import enhanced_index_document, EnhancedDocumentProcessor
from enhanced_retriever import enhanced_search_documents

def main():
    st.set_page_config(
        page_title="Enhanced Edge RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI and Arabic support
    st.markdown("""
    <style>
    .rtl { direction: rtl; text-align: right; font-family: 'Arial', sans-serif; }
    .doc-type-tag { 
        background-color: #e1f5fe; 
        color: #01579b; 
        padding: 2px 8px; 
        border-radius: 12px; 
        font-size: 12px; 
        margin: 2px;
        display: inline-block;
    }
    .confidence-high { color: #2e7d32; font-weight: bold; }
    .confidence-medium { color: #f57c00; font-weight: bold; }
    .confidence-low { color: #d32f2f; font-weight: bold; }
    .importance-score { 
        background: linear-gradient(90deg, #ff9800, #4caf50); 
        color: white; 
        padding: 2px 6px; 
        border-radius: 8px; 
        font-size: 11px;
    }
    .match-reason {
        background-color: #f3e5f5;
        color: #4a148c;
        padding: 1px 6px;
        border-radius: 6px;
        font-size: 10px;
        margin: 1px;
        display: inline-block;
    }
    .source-header {
        background-color: #f5f5f5;
        padding: 8px;
        border-left: 4px solid #2196f3;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🧠 Enhanced Edge RAG System")
    st.markdown("*Advanced Document Processing with Intelligent Chunking & Context-Aware Retrieval*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Enhanced Configuration")
        
        # Model Selection
        st.subheader("🤖 Models")
        embed_model = st.selectbox(
            "Embedding Model",
            ["bge-m3", "qwen2.5:0.5b", "gemma3:1b"],
            index=0,
            help="BGE-M3 provides best multilingual performance"
        )
        
        generation_model = st.selectbox(
            "Generation Model",
            ["qwen2.5:0.5b", "gemma3:1b", "llama3.2:1b"],
            index=0,
            help="Qwen2.5 provides excellent Arabic support"
        )
        
        # Search Configuration
        st.subheader("🔍 Search Settings")
        doc_type_filter = st.selectbox(
            "Document Type Filter",
            ["All", "business_partnership", "legal_arabic", "events_business", "form_document"],
            index=0,
            help="Filter by specific document types for more targeted results"
        )
        
        importance_threshold = st.slider(
            "Minimum Importance Score",
            0.0, 1.0, 0.0, 0.1,
            help="Filter chunks by importance (0.7+ for high-importance content only)"
        )
        
        max_results = st.slider(
            "Maximum Results",
            3, 15, 8,
            help="Number of chunks to retrieve and analyze"
        )
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            show_debug_info = st.checkbox("Show Debug Information", value=False)
            show_chunk_details = st.checkbox("Show Chunk Details", value=True)
            show_match_reasons = st.checkbox("Show Match Reasoning", value=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    # Document Upload and Processing
    with col1:
        st.header("📄 Enhanced Document Processing")
        
        # File Upload
        uploaded_files = st.file_uploader(
            "Upload documents for intelligent processing",
            type=['txt', 'pdf', 'json', 'csv'],
            accept_multiple_files=True,
            help="Supports text files and PDFs with automatic type detection"
        )
        
        if uploaded_files:
            if st.button("🚀 Process with Enhanced Indexing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                processor = EnhancedDocumentProcessor()
                processing_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Save uploaded file temporarily
                        temp_dir = "temp"
                        os.makedirs(temp_dir, exist_ok=True)
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Detect document type
                        if uploaded_file.type == "application/pdf":
                            content = uploaded_file.getbuffer()
                        else:
                            content = uploaded_file.getvalue().decode("utf-8")
                        
                        doc_profile = processor.detect_document_type(str(content)[:1000], uploaded_file.name)
                        
                        status_text.text(f"Processing {uploaded_file.name} as {doc_profile.name} document...")
                        
                        # Enhanced indexing
                        success = enhanced_index_document(
                            file_path,
                            content,
                            metadata={"uploaded_via": "streamlit_enhanced", "upload_time": datetime.now().isoformat()},
                            embed_model=embed_model
                        )
                        
                        processing_results.append({
                            'filename': uploaded_file.name,
                            'doc_type': doc_profile.name,
                            'success': success,
                            'chunk_size': doc_profile.chunk_size,
                            'overlap_ratio': doc_profile.overlap_ratio
                        })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Clean up temp file
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                    except Exception as e:
                        processing_results.append({
                            'filename': uploaded_file.name,
                            'success': False,
                            'error': str(e)
                        })
                
                # Display processing results
                with results_container:
                    st.subheader("📋 Processing Results")
                    for result in processing_results:
                        if result['success']:
                            st.success(
                                f"✅ **{result['filename']}** processed successfully\\n"
                                f"📊 Type: `{result['doc_type']}` | "
                                f"📏 Chunk Size: `{result['chunk_size']}` | "
                                f"🔄 Overlap: `{result['overlap_ratio']:.1%}`"
                            )
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"❌ **{result['filename']}** failed: {error_msg}")
                
                status_text.success("🎉 Enhanced processing complete!")
        
        # Display Currently Indexed Files
        st.subheader("📚 Document Index Status")
        
        # Try to get file info from API
        try:
            response = requests.get("http://localhost:8000/indexed-files", timeout=2)
            if response.status_code == 200:
                files_data = response.json().get("files", [])
                
                if files_data:
                    for file_info in files_data:
                        with st.expander(f"📄 {file_info['filename']}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Language:** {file_info['language']}")
                                st.write(f"**Chunks:** {file_info['chunks']}")
                            with col_b:
                                if 'doc_type' in file_info:
                                    st.markdown(f'<span class="doc-type-tag">{file_info["doc_type"]}</span>', 
                                              unsafe_allow_html=True)
                else:
                    st.info("No indexed files found via API.")
            else:
                st.warning("Could not connect to backend API.")
        except:
            st.info("📁 Files uploaded via this interface are processed directly.")
    
    # Enhanced Query Interface
    with col2:
        st.header("🔍 Enhanced Query & Retrieval")
        
        # Query Input
        query = st.text_area(
            "Enter your question (English/Arabic)",
            placeholder="Ask about your documents... اسأل عن وثائقك...",
            height=100,
            help="The system will automatically detect language and optimize search accordingly"
        )
        
        # Query Examples
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **English Examples:**
            - What is the Microsoft G42 partnership about?
            - How much did Microsoft invest in G42?
            - What are the key aspects of the Azure collaboration?
            
            **Arabic Examples:**
            - ما هي شراكة مايكروسوفت مع G42؟
            - كم استثمرت مايكروسوفت في الشراكة؟
            - ما هي المواد المهمة في القانون؟
            """)
        
        if st.button("🔍 Enhanced Search & Generate", type="primary") and query:
            with st.spinner("🧠 Processing with enhanced intelligence..."):
                try:
                    # Prepare search parameters
                    search_doc_type = None if doc_type_filter == "All" else doc_type_filter
                    
                    # Perform enhanced search
                    result = enhanced_search_documents(
                        query=query,
                        embed_model=embed_model,
                        generation_model=generation_model,
                        doc_type=search_doc_type,
                        importance_threshold=importance_threshold,
                        max_results=max_results
                    )
                    
                    # Display Enhanced Response
                    st.subheader("💡 Enhanced AI Response")
                    
                    # Confidence indicator
                    confidence = result.get('confidence', 0.0)
                    if confidence >= 0.8:
                        conf_class = "confidence-high"
                        conf_icon = "🟢"
                    elif confidence >= 0.5:
                        conf_class = "confidence-medium"
                        conf_icon = "🟡"
                    else:
                        conf_class = "confidence-low"
                        conf_icon = "🔴"
                    
                    st.markdown(f"**Confidence:** {conf_icon} <span class='{conf_class}'>{confidence:.1%}</span> | "
                              f"**Context Quality:** {result.get('context_quality', 'unknown').title()}", 
                              unsafe_allow_html=True)
                    
                    # Display response with RTL support for Arabic
                    response_text = result.get('response', '')
                    if result.get('search_stats', {}).get('query_language') == 'arabic':
                        st.markdown(f'<div class="rtl">{response_text}</div>', unsafe_allow_html=True)
                    else:
                        st.write(response_text)
                    
                    # Enhanced Source Information
                    st.subheader("📖 Enhanced Source Analysis")
                    
                    sources = result.get('sources', [])
                    if sources:
                        # Search Statistics
                        stats = result.get('search_stats', {})
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Sources Found", stats.get('total_results', 0))
                        with col_stat2:
                            st.metric("Sources Used", stats.get('sources_used', 0))
                        with col_stat3:
                            st.metric("Avg Importance", f"{stats.get('avg_importance', 0):.2f}")
                        
                        # Detailed Source Information
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"📄 Source {i}: {source['source']} ({source['doc_type']})"):
                                # Source header with metadata
                                st.markdown(f"""
                                <div class="source-header">
                                    <strong>Document:</strong> {source['source']} | 
                                    <strong>Type:</strong> <span class="doc-type-tag">{source['doc_type']}</span> | 
                                    <strong>Semantic:</strong> {source['semantic_type']}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Content
                                st.write("**Content:**")
                                if result.get('search_stats', {}).get('query_language') == 'arabic':
                                    st.markdown(f'<div class="rtl">{source["text"]}</div>', unsafe_allow_html=True)
                                else:
                                    st.write(source['text'])
                                
                                # Scoring Information
                                if show_chunk_details:
                                    st.write("**Scoring Details:**")
                                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                    
                                    with col_s1:
                                        st.metric("Similarity", f"{source['similarity_score']:.3f}")
                                    with col_s2:
                                        st.metric("Final Score", f"{source['final_score']:.3f}")
                                    with col_s3:
                                        importance = source['importance_score']
                                        st.markdown(f"**Importance:** <span class='importance-score'>{importance:.2f}</span>", 
                                                  unsafe_allow_html=True)
                                    with col_s4:
                                        st.metric("Relevance×", f"{source['relevance_multiplier']:.2f}")
                                
                                # Chunk Information
                                if show_chunk_details:
                                    chunk_info = source.get('chunk_info', {})
                                    st.write("**Chunk Details:**")
                                    st.write(f"• Chunk {chunk_info.get('chunk_id', 0) + 1} of {chunk_info.get('total_chunks', 1)}")
                                    st.write(f"• Word count: {chunk_info.get('word_count', 0)}")
                                    st.write(f"• Has structure: {'Yes' if chunk_info.get('has_structure', False) else 'No'}")
                                    
                                    if chunk_info.get('structural_markers'):
                                        st.write(f"• Structural markers: {', '.join(chunk_info['structural_markers'])}")
                                
                                # Match Reasoning
                                if show_match_reasons and source.get('match_reasons'):
                                    st.write("**Why this matched:**")
                                    for reason in source['match_reasons']:
                                        st.markdown(f'<span class="match-reason">{reason}</span>', 
                                                  unsafe_allow_html=True)
                                
                                # Debug Information
                                if show_debug_info:
                                    st.write("**Debug Info:**")
                                    st.json({
                                        'semantic_type': source['semantic_type'],
                                        'doc_type': source['doc_type'],
                                        'scores': {
                                            'similarity': source['similarity_score'],
                                            'final': source['final_score'],
                                            'importance': source['importance_score'],
                                            'relevance_multiplier': source['relevance_multiplier']
                                        }
                                    })
                    else:
                        st.warning("No relevant sources found. Try rephrasing your question or adjusting the filters.")
                    
                except Exception as e:
                    st.error(f"🚨 Enhanced search error: {str(e)}")
                    if show_debug_info:
                        st.exception(e)
                    st.error("Make sure the enhanced indexing has been completed and backend services are running.")

# Footer with information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
🧠 Enhanced Edge RAG System | 
Intelligent Document Processing | 
Context-Aware Retrieval | 
Multilingual Support (Arabic/English)
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
