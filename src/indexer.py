import os, uuid, json, csv, textwrap, requests, ollama, numpy as np, tempfile
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

load_dotenv()

# Azure services
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY      = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_DOC_INTEL_ENDPOINT= os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTEL_KEY     = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# Qdrant
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY= os.getenv("QDRANT_API_KEY", "")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

# Optional default models from .env (UI can override)
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# ---------- Azure clients ----------
try:
    di_client = DocumentAnalysisClient(AZURE_DOC_INTEL_ENDPOINT, AzureKeyCredential(AZURE_DOC_INTEL_KEY))
    HAS_DI = True
except Exception as e:
    print(f"Warn: Document Intelligence unavailable: {e}")
    HAS_DI = False
    di_client = None

def _model_slug(m: str) -> str:
    return m.replace("/", "_").replace(":", "_")

def _bge(model: str) -> bool:
    return "bge" in model.lower()

def ensure_collection(client: QdrantClient, language: str, embed_model: str, first_vector: List[float]) -> str:
    lang = "ar" if language == "arabic" else "en"
    name = f"rag_docs_{lang}_{_model_slug(embed_model)}"
    size = len(first_vector)
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if name not in collection_names:
        client.create_collection(name, vectors_config=VectorParams(size=size, distance="Cosine"))
        print(f"Created collection '{name}' (size={size})")
    return name

def detect_language(text: str) -> str:
    try:
        base = AZURE_LANGUAGE_ENDPOINT.rstrip("/")
        r = requests.post(
            f"{base}/text/analytics/v3.1/languages",
            headers={"Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY, "Content-Type": "application/json"},
            json={"documents":[{"id":"1","text":text}]}
        )
        r.raise_for_status()
        docs = r.json().get("documents",[])
        if docs:
            return "arabic" if docs[0]["detectedLanguage"]["iso6391Name"]=="ar" else "english"
    except Exception as e:
        print(f"Lang detect err: {e}")
    return "english"

def generate_embedding(text: str, language: str, model: str, role: str="passage") -> List[float]:
    prompt = f"{role}: {text}" if _bge(model) else text
    resp = ollama.embeddings(model=model, prompt=prompt)
    return resp["embedding"]

def extract_entities(text: str, language: str) -> List[Dict[str,str]]:
    try:
        base = AZURE_LANGUAGE_ENDPOINT.rstrip("/")
        r = requests.post(
            f"{base}/text/analytics/v3.1/entities/recognition/general",
            headers={"Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY, "Content-Type":"application/json"},
            json={"documents":[{"id":"1","text":text,"language":"ar" if language=="arabic" else "en"}]}
        )
        r.raise_for_status()
        docs = r.json().get("documents",[])
        if docs:
            return [{"text":e["text"],"category":e["category"]} for e in docs[0].get("entities",[])]
    except Exception as e:
        print(f"Entities err: {e}")
    return []

def extract_key_phrases(text: str, language: str) -> List[str]:
    try:
        client_ta = TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=AzureKeyCredential(AZURE_LANGUAGE_KEY))
        lang = "ar" if language=="arabic" else "en"
        out=[]
        for chunk in [text[i:i+5000] for i in range(0,len(text),5000)]:
            res = client_ta.extract_key_phrases([chunk], language=lang)[0]
            if not res.is_error: out.extend(res.key_phrases)
        return out
    except Exception as e:
        print(f"Key phrases err: {e}")
        return []

def analyze_sentiment(text: str, language: str) -> Dict[str,float]:
    try:
        client_ta = TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=AzureKeyCredential(AZURE_LANGUAGE_KEY))
        lang = "ar" if language=="arabic" else "en"
        agg={"positive":0.0,"neutral":0.0,"negative":0.0}; n=0
        for chunk in [text[i:i+5000] for i in range(0,len(text),5000)]:
            res=client_ta.analyze_sentiment([chunk],language=lang)[0]
            if not res.is_error:
                agg["positive"]+=res.confidence_scores.positive
                agg["neutral"] +=res.confidence_scores.neutral
                agg["negative"]+=res.confidence_scores.negative
                n+=1
        if n:
            for k in agg: agg[k]/=n
        return agg
    except Exception as e:
        print(f"Sentiment err: {e}")
        return {"positive":0.0,"neutral":0.0,"negative":0.0}

def _pages_from_di_result(result):

    # Build page buckets

    pages = {p.page_number: [] for p in result.pages}

    if getattr(result, "paragraphs", None):

        for para in result.paragraphs:

            if para.bounding_regions:

                pg = para.bounding_regions[0].page_number

                pages.setdefault(pg, []).append(para.content or "")

    else:

        for p in result.pages:

            pg = p.page_number

            lines = []

            if getattr(p, "lines", None):

                for ln in p.lines:

                    words = " ".join([w.content for w in (ln.words or [])]).strip()

                    if words:

                        lines.append(words)

            pages[pg] = lines

    return [{"page": pg, "text": "\n".join([t for t in pages[pg] if t])}

            for pg in sorted(pages)]

 
def process_with_document_intelligence(file_path: str) -> Dict[str,Any]:
    import PyPDF2
    # 1. Try Azure Document Intelligence if available
    if HAS_DI:
        try:
            with open(file_path,"rb") as f:
                poller = di_client.begin_analyze_document("prebuilt-document", f)
            res = poller.result()
            out = {"text": res.content, "tables":[], "key_value_pairs":[], "entities":[], "metadata": {"page_count": len(res.pages)}}
            for t in res.tables:
                table={"row_count":t.row_count,"column_count":t.column_count,"cells":[]}
                for c in t.cells: table["cells"].append({"text":c.content,"row_index":c.row_index,"column_index":c.column_index})
                out["tables"].append(table)
            for kv in res.key_value_pairs:
                if kv.key and kv.value: out["key_value_pairs"].append({"key":kv.key.content,"value":kv.value.content,"confidence":kv.confidence})
            for e in res.entities:
                out["entities"].append({"text":e.text,"category":e.category,"confidence":e.confidence})
            return out
        except Exception as e:
            print(f"[PDF] Azure DI failed: {e}")
    # 2. Try PyPDF2 for text extraction
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        print(f"[PDF] PyPDF2 extracted {len(text)} characters from '{file_path}'")
        return {"text": text, "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}
    except Exception as e:
        print(f"[PDF] PyPDF2 extract err: {e}")
    # 3. Last resort: decode as bytes (may be garbage)
    try:
        with open(file_path,"rb") as f:
            raw = f.read().decode("utf-8",errors="ignore")
        print(f"[PDF] WARNING: Decoding raw bytes for '{file_path}', result may be garbage!")
        return {"text": raw, "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}
    except Exception as e:
        print(f"[PDF] Final fallback failed: {e}")
        return {"text": "", "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}
        import PyPDF2
        if not HAS_DI:
            # Fallback: Try PyPDF2 for text extraction
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return {"text": text, "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}
            except Exception as e:
                print(f"PDF extract err: {e}")
                return {"text": "", "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}
        try:
            with open(file_path,"rb") as f:
                poller = di_client.begin_analyze_document("prebuilt-document", f)
            res = poller.result()
            out = _pages_from_di_result(res)
            for t in res.tables:
                table={"row_count":t.row_count,"column_count":t.column_count,"cells":[]}
                for c in t.cells: table["cells"].append({"text":c.content,"row_index":c.row_index,"column_index":c.column_index})
                out["tables"].append(table)
            for kv in res.key_value_pairs:
                if kv.key and kv.value: out["key_value_pairs"].append({"key":kv.key.content,"value":kv.value.content,"confidence":kv.confidence})
            for e in res.entities:
                out["entities"].append({"text":e.text,"category":e.category,"confidence":e.confidence})
            return out
        except Exception as e:
            print(f"DI err: {e}")
            # Fallback: Try PyPDF2 for text extraction
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return {"text": text, "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}
            except Exception as e2:
                print(f"PDF extract err: {e2}")
                return {"text": "", "tables":[], "key_value_pairs":[], "entities":[], "metadata":{}}

def index_document(file_path: str, content: Union[str,bytes], metadata: Dict[str,Any]=None, embed_model: str=DEFAULT_EMBED_MODEL) -> bool:
    try:
        metadata = metadata or {}
        if "source" not in metadata: metadata["source"]=os.path.basename(file_path)

        # get text (PDF via DI, others directly)
        if file_path.lower().endswith(".pdf"):
            # Always use full path for PDFs in data folder
            full_path = os.path.join("data", file_path) if not os.path.isabs(file_path) else file_path
            di = process_with_document_intelligence(full_path)
            text = di["text"]
            print(f"[INDEXER DEBUG] Extracted text length from PDF '{file_path}': {len(text) if text else 0}")
            metadata.update(di)
        else:
            text = content.decode("utf-8",errors="ignore") if isinstance(content,bytes) else str(content)

        chunks = textwrap.wrap(text, CHUNK_SIZE) if text.strip() else []
        for i, chunk in enumerate(chunks):
            lang = detect_language(chunk)
            ents_raw = extract_entities(chunk, lang)
            ents_by_cat={}
            for e in ents_raw:
                ents_by_cat.setdefault(e["category"],[]).append(e["text"])
            phrases = extract_key_phrases(chunk, lang)
            senti   = analyze_sentiment(chunk, lang)

            vec = generate_embedding(chunk, lang, model=embed_model, role="passage")
            coll = ensure_collection(client, lang, embed_model, vec)
            print(f"[INDEXER DEBUG] Using collection: {coll}")

            payload = {
                "text": chunk,
                "source": metadata.get("source"),
                "language": lang,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "entities": ents_by_cat,
                "key_phrases": phrases,
                "sentiment": senti,
                "metadata": metadata
            }
            client.upsert(collection_name=coll, points=[PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)])
        print("Indexed OK")
        return True
    except Exception as e:
        print(f"Index error: {e}")
        return False

def load_documents() -> List[Dict[str,Any]]:
    docs=[]; folder="data"
    if not os.path.isdir(folder): return docs
    for fn in os.listdir(folder):
        p=os.path.join(folder,fn)
        if fn.endswith(".txt"):
            with open(p,"r",encoding="utf-8") as f:
                t=f.read()
                if t.strip(): docs.append({"text":t,"filename":fn})
        elif fn.endswith(".json"):
            with open(p,"r",encoding="utf-8") as f:
                for d in json.load(f):
                    if d.get("text","").strip(): docs.append({"text":d["text"],"filename":fn})
        elif fn.endswith(".csv"):
            with open(p,"r",encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("text","").strip(): docs.append({"text":r["text"],"filename":fn})
        elif fn.endswith(".pdf"):
            with open(p, "rb") as f:
                content = f.read()
                if content:
                    docs.append({"text": content, "filename": fn})
    return docs

if __name__=="__main__":
    ds=load_documents()
    if not ds:
        print("No docs in data/"); exit()
    for i,d in enumerate(ds,1):
        print(f"Indexing {i}/{len(ds)}: {d['filename']}")
        index_document(d["filename"], d["text"])
    print("Done.")
 