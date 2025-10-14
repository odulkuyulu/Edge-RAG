import os, tempfile, streamlit as st, nltk, ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from indexer import index_document, load_documents
from retriever import search_documents, generate_response, detect_language_azure

load_dotenv()

QDRANT_URL=os.getenv("QDRANT_URL","http://localhost:6333")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY","")
client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

st.set_page_config(page_title="Edge RAG", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

def setup():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt', quiet=True)
setup()

with st.sidebar:
    st.title("📄 Document Management")

    # Model pickers
    st.subheader("Models")
    EMBED_MODELS=["bge-m3","gemma2:2b","qwen2.5:0.5b","jaluma/arabert-all-nli-triplet-matryoshka:latest"]
    LLM_MODELS  =["gemma3:1b","phi4-mini:3.8b","qwen2.5:3b","llama3.1:8b","qwen2.5:7b-instruct"]
    embed_model = st.selectbox("Embedding model", EMBED_MODELS, index=0, key="embed_model")
    gen_model   = st.selectbox("LLM (generation)", LLM_MODELS, index=0, key="gen_model")
    st.caption("Switch models any time. Re-index if you change the embedding model.")

    st.subheader("Upload Documents")
    up = st.file_uploader("Choose a file", type=["txt","pdf"])
    if up:
        try:
            with st.spinner("Indexing..."):
                # Read content (bytes for pdf, str for txt)
                if up.type=="application/pdf":
                    content = up.read()  # bytes
                else:
                    content = up.read().decode("utf-8")  # str

                # Write a temp file (match type when writing)
                suffix = os.path.splitext(up.name)[1] or ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    if isinstance(content,str):
                        tf.write(content.encode("utf-8"))   # <-- FIX: write bytes
                    else:
                        tf.write(content)                   # bytes for PDFs
                    temp_path=tf.name
                try:
                    ok = index_document(temp_path, content, {"source": up.name}, embed_model=embed_model)
                    st.success("✅ Document indexed!" if ok else "⚠️ Indexing failed.")
                finally:
                    os.unlink(temp_path)
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("📂 Load Sample Documents"):
        with st.spinner("Loading..."):
            try:
                for d in load_documents():
                    index_document(d["filename"], d["text"], {"source": d["filename"]}, embed_model=embed_model)
                st.success("Sample docs loaded.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- Vector DB: Qdrant\n- Embeddings: selectable\n- LLM: selectable\n- Azure Language: NER / KP / Sentiment")

st.markdown("<h1 style='text-align:center'>🔍 Edge RAG Search</h1>", unsafe_allow_html=True)
query = st.text_input("Ask a question:", placeholder="e.g., Who announced their strategic partnership in UAE?")

@st.cache_data
def _cached_search(q:str, lang:str, em:str, gm:str):
    return search_documents(q, lang, embed_model=em, gen_model=gm)

@st.cache_data
def _cached_response(q:str, results:list, gm:str):
    return generate_response(q, results, gen_model=gm)

if st.button("🔎 Search", use_container_width=True):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Detecting language & searching..."):
            lang=detect_language_azure(query).get("language","english")
            st.info(("🇦🇪 Arabic" if lang=="arabic" else "🇺🇸 English")+" detected")

            results = _cached_search(query, lang, embed_model, gen_model)
            if results:
                resp = _cached_response(query, results, gen_model)
                st.subheader("🤖 AI Response")
                if lang=="arabic":
                    st.markdown(f"<div dir='rtl' style='text-align:right'>{resp}</div>", unsafe_allow_html=True)
                else:
                    st.write(resp)

                st.subheader("📚 Sources")
                for i,r in enumerate(results,1):
                    with st.expander(f"Source {i} (Relevance: {r.get('score',0):.2f})"):
                        st.write(f"**Document:** {os.path.basename(r.get('source','Unknown'))}")
                        st.write(f"**Language:** {r.get('language','unknown').capitalize()}")
                        st.write("**Relevant Content:**")
                        if r.get("language")=="arabic":
                            st.markdown(f"<div dir='rtl' style='text-align:right'>{r.get('text','')}</div>", unsafe_allow_html=True)
                        else:
                            st.write(r.get("text",""))
                        if r.get("matched_entities"):
                            st.write("**Matched Entities:**")
                            for k,v in r["matched_entities"].items():
                                st.write(f"- {k}: {', '.join(v)}")
            else:
                st.warning("No relevant documents found.")
 