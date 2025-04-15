import streamlit as st
from retriever import generate_response, search_documents, detect_language
from indexer import index_document, load_documents, create_collection_if_not_exists
import ollama
import os
import nltk
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def setup_app():
    """Initialize app dependencies and configurations."""
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Initialize Qdrant collections and load sample documents
    try:
        # Create collections if they don't exist
        create_collection_if_not_exists(client, "rag_docs_en")
        create_collection_if_not_exists(client, "rag_docs_ar")
        
        # Load sample documents if not already loaded
        if not st.session_state.get('documents_loaded', False):
            with st.spinner("Loading sample documents..."):
                load_documents()
                st.session_state.documents_loaded = True
                st.success("✅ Sample documents loaded successfully!")
    except Exception as e:
        st.error(f"Error initializing collections: {str(e)}")

# 🎨 Streamlit UI Setup
st.set_page_config(
    page_title="Edge RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run setup
setup_app()

# 🌍 Header
st.markdown("<h1 style='text-align: center;'>🔍 Edge RAG Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Azure AI Services to enhance accuracy, retrieval, and insights.</p>", unsafe_allow_html=True)

# Initialize session state
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar
with st.sidebar:
    st.title("📄 Document Management")
    
    # Document Upload Section
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf'],
        help="Supported formats: TXT, PDF"
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                st.write("Reading file content...")
                if uploaded_file.type == "application/pdf":
                    # For PDF files, read as binary
                    content = uploaded_file.read()
                    st.write("PDF content read successfully")
                else:
                    # For text files, read as text
                    content = uploaded_file.read().decode('utf-8')
                    st.write("Text content read successfully")
                
                st.write("Indexing document...")
                index_document(content, uploaded_file.name)
                st.success("✅ Document indexed successfully!")
                st.session_state.documents_indexed = True
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.error("Please check if Azure Document Intelligence service is running and properly configured.")

    # Load Documents Button
    if st.button("📂 Load Sample Documents", help="Load pre-indexed sample documents"):
        with st.spinner("Loading documents..."):
            try:
                load_documents()
                st.success("✅ Documents loaded successfully!")
                st.session_state.documents_indexed = True
            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")
    
    # Tech Stack Information
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🔍 **Vector DB**: 
        - Qdrant
    - 🧠 **Embeddings**: 
        - bge-m3
    - 🤖 **LLM**: 
        - Arabic: gemma3:1b
        - English: phi4-mini:3.8b
    - ⚡ **Azure AI Disconnected Containers**: 
        - Language Detection 
        - Named Entity Recognition (NER) 
        - Text Analytics 
    """)
    
    # Language Support
    st.markdown("### 🌍 Language Support")
    st.markdown("""
    - **English** 🇺🇸
    - **Arabic** 🇦🇪
    """)

# Main Content

# Search Input
query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What did Microsoft and OpenAI announce?",
    key="search_input"
)

@st.cache_resource
def get_ollama_model():
    """Get cached Ollama model instance."""
    return ollama

@st.cache_data
def cached_search(query: str, language: str):
    """Cache search results to prevent recomputation."""
    return search_documents(query, language)

@st.cache_data
def cached_response(query: str, results: list):
    """Cache AI responses to prevent recomputation."""
    try:
        # Ensure results is a list of dictionaries
        if not isinstance(results, list):
            results = []
        if results and not isinstance(results[0], dict):
            results = []
        
        return generate_response(query, results)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but there was an error processing your request."

# Search Button
if st.button("🔍 Search", use_container_width=True, disabled=st.session_state.is_loading):
    if query:
        st.session_state.is_loading = True
        try:
            with st.spinner("🔍 Searching through documents..."):
                # Detect language automatically
                language = detect_language(query)
                
                # Show detected language with appropriate emoji
                lang_emoji = "🇺🇸" if language == "english" else "🇦🇪"
                lang_display = "English" if language == "english" else "Arabic"
                st.info(f"{lang_emoji} Detected Language: {lang_display}")
                
                # Search for relevant documents
                results = cached_search(query, language)
                
                if results:
                    with st.spinner("🤖 Generating response..."):
                        response = cached_response(query, results)
                        st.subheader("🤖 AI Response")
                        
                        # Add RTL support for Arabic responses
                        if language == "arabic":
                            st.markdown(f"""
                            <div dir="rtl" style="text-align: right; font-family: 'Arial', sans-serif; line-height: 1.8; background-color: #1E1E1E; color: #FFFFFF; padding: 20px; border-radius: 10px; border: 1px solid #333;">
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Add custom styling without source formatting
                            st.markdown(f"""
                            <div style="font-family: 'Arial', sans-serif; line-height: 1.8; background-color: #1E1E1E; color: #FFFFFF; padding: 20px; border-radius: 10px; border: 1px solid #333;">
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                        
                    # Display sources below the response
                    st.subheader("📚 Sources")
                    for i, doc in enumerate(results, 1):
                        if not isinstance(doc, dict):
                            continue
                            
                        source_name = doc.get('source', 'Unknown').split('/')[-1] if doc.get('source') else 'Unknown'
                        lang_emoji = "🇺🇸" if doc.get('language') == "english" else "🇦🇪"
                        lang_display = doc.get('language', 'unknown').capitalize()
                        
                        # Add RTL support for Arabic source content
                        with st.expander(f"Source {i} (Relevance: {doc.get('score', 0):.2f})"):
                            st.write(f"**Document:** {source_name}")
                            st.write(f"**Language:** {lang_emoji} {lang_display}")
                            st.write("**Relevant Content:**")
                            
                            if doc.get('text'):
                                if doc.get('language') == "arabic":
                                    st.markdown(f"""
                                    <div dir="rtl" style="text-align: right; font-family: 'Arial', sans-serif; line-height: 1.8;">
                                        {doc['text']}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.write(doc['text'])
                            
                            # Display matched entities if available
                            matched_entities = doc.get('matched_entities', {})
                            if matched_entities and isinstance(matched_entities, dict):
                                st.write("**Matched Entities:**")
                                for category, entities in matched_entities.items():
                                    if isinstance(entities, list):
                                        st.write(f"- {category}: {', '.join(entities)}")
                                    else:
                                        st.write(f"- {category}: {entities}")
                else:
                    st.warning("No relevant documents found for your query.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.is_loading = False
    else:
        st.warning("Please enter a question to search.")

# Example Prompts Section at the bottom
st.markdown("---")
st.markdown("### 📝 Test Prompts for Indexed Data")

# English Prompts
st.markdown("#### 🇺🇸 English Prompts")
st.code("""
# Test these prompts with our indexed data:

1. "What is the value of Microsoft and OpenAI's partnership?"
   - Tests entity recognition and numerical information
   - Expected: $10 billion partnership information

2. "What are Amazon's AWS plans for healthcare and finance?"
   - Tests specific service information
   - Expected: Machine learning services details
""", language="markdown")

# Arabic Prompts
st.markdown("#### 🇦🇪 Arabic Prompts")
st.code("""
# جرب هذه الأسئلة مع البيانات المفهرسة:

١. "ما هي قيمة شراكة مايكروسوفت وG42 في الإمارات؟"
   - يختبر التعرف على الكيانات والمعلومات الرقمية
   - المتوقع: معلومات عن شراكة بقيمة ١.٥ مليار دولار

٢. "ما هي خطط الذكاء الاصطناعي في أبوظبي؟"
   - يختبر معلومات عن المشاريع المحلية
   - المتوقع: معلومات عن مركز الذكاء الاصطناعي الجديد
""", language="markdown")

# Add a note about the data
st.info("""
ℹ️ These prompts are specifically designed to test the indexed data in our system.
The responses should include information from our sample documents about:
- Microsoft and OpenAI's $10 billion partnership
- Amazon's AWS machine learning services
- Microsoft and G42's $1.5 billion partnership in UAE
- AI developments in Abu Dhabi
""")