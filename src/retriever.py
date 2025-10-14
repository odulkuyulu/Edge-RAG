import os, json, requests, ollama, numpy as np, nltk, logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from functools import lru_cache
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

import os
import nltk

nltk_data_path = "C:/Users/odulkuyulu/nltk_data"
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path.append(nltk_data_path)

load_dotenv()
nltk.download("punkt_tab", quiet=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Qdrant
QDRANT_URL=os.getenv("QDRANT_URL","http://localhost:6333"); QDRANT_API_KEY=os.getenv("QDRANT_API_KEY","")
client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Azure Language
AZURE_LANGUAGE_ENDPOINT=os.getenv("AZURE_LANGUAGE_ENDPOINT"); AZURE_LANGUAGE_KEY=os.getenv("AZURE_LANGUAGE_KEY")

# Defaults (UI can override)
DEFAULT_EMBED_MODEL=os.getenv("EMBEDDING_MODEL","qwen2.5:0.5b")
DEFAULT_GEN_MODEL  =os.getenv("GENERATION_MODEL","gemma3:1b")

def _slug(m:str)->str: return m.replace("/", "_").replace(":", "_")
def _bge(m:str)->bool: return "bge" in m.lower()

def detect_language_azure(text:str)->Dict[str,Any]:
    try:
        url=f"{AZURE_LANGUAGE_ENDPOINT.rstrip('/')}/language/:analyze-text?api-version=2023-04-01"
        r=requests.post(url,headers={"Content-Type":"application/json","Ocp-Apim-Subscription-Key":AZURE_LANGUAGE_KEY},
                        json={"kind":"LanguageDetection","analysisInput":{"documents":[{"id":"1","text":text}]}})
        r.raise_for_status()
        docs=r.json().get("results",{}).get("documents",[])
        if docs:
            iso=docs[0]["detectedLanguage"]["iso6391Name"]
            return {"language": "arabic" if iso=="ar" else "english", "confidence": docs[0]["detectedLanguage"]["confidenceScore"]}
    except Exception as e:
        log.error(f"Lang detect err: {e}")
    return {"language":"english","confidence":0.0}

def extract_entities_azure(text:str, language:str)->Dict[str,List[str]]:
    try:
        url=f"{AZURE_LANGUAGE_ENDPOINT.rstrip('/')}/language/:analyze-text?api-version=2023-04-01"
        r=requests.post(url,headers={"Content-Type":"application/json","Ocp-Apim-Subscription-Key":AZURE_LANGUAGE_KEY},
                        json={"kind":"EntityRecognition","analysisInput":{"documents":[{"id":"1","text":text,"language":"ar" if language=="arabic" else "en"}]}})
        r.raise_for_status()
        out={}
        for doc in r.json().get("results",{}).get("documents",[]):
            for e in doc.get("entities",[]):
                out.setdefault(e["category"],[])
                if e["text"] not in out[e["category"]]: out[e["category"]].append(e["text"])
        return out
    except Exception as e:
        log.error(f"Entities err: {e}"); return {}

def extract_key_phrases_azure(text:str, language:str)->List[str]:
    try:
        url=f"{AZURE_LANGUAGE_ENDPOINT.rstrip('/')}/language/:analyze-text?api-version=2023-04-01"
        r=requests.post(url,headers={"Content-Type":"application/json","Ocp-Apim-Subscription-Key":AZURE_LANGUAGE_KEY},
                        json={"kind":"KeyPhraseExtraction","analysisInput":{"documents":[{"id":"1","text":text,"language":"ar" if language=="arabic" else "en"}]}})
        r.raise_for_status()
        phrases=[]
        for doc in r.json().get("results",{}).get("documents",[]): phrases.extend(doc.get("keyPhrases",[]))
        return phrases
    except Exception as e:
        log.error(f"Keyphrases err: {e}"); return []
def analyze_sentiment_azure(text:str, language:str)->Dict[str,float]:
    try:
        url=f"{AZURE_LANGUAGE_ENDPOINT.rstrip('/')}/language/:analyze-text?api-version=2023-04-01"
        r=requests.post(url,headers={"Content-Type":"application/json","Ocp-Apim-Subscription-Key":AZURE_LANGUAGE_KEY},
                        json={"kind":"SentimentAnalysis","analysisInput":{"documents":[{"id":"1","text":text,"language":"ar" if language=="arabic" else "en"}]}})
        r.raise_for_status()
        for doc in r.json().get("results",{}).get("documents",[]):
            cs=doc.get("confidenceScores")
            if cs: return {"positive":cs["positive"],"neutral":cs["neutral"],"negative":cs["negative"]}
    except Exception as e:
        log.error(f"Sentiment err: {e}")
    return {"positive":0.0,"neutral":0.0,"negative":0.0}

@lru_cache(maxsize=1000)
def _embed(text:str, language:str, model:str, role:str)->List[float]:
    prompt = f"{role}: {text}" if _bge(model) else text
    resp = ollama.embeddings(model=model, prompt=prompt)
    return resp["embedding"]

def search_documents(query:str, language:str=None, embed_model:str=DEFAULT_EMBED_MODEL, gen_model:str=DEFAULT_GEN_MODEL)->List[Dict[str,Any]]:
    try:
        if not language:
            language = detect_language_azure(query)["language"]
        lang_code = "ar" if language=="arabic" else "en"

        q_entities = extract_entities_azure(query, language)
        q_phrases  = extract_key_phrases_azure(query, language)
        q_sent     = analyze_sentiment_azure(query, language)

        q_vec = _embed(query, language, model=embed_model, role="query")

        # Only search the model-specific collection for the detected language
        coll_name = f"rag_docs_{lang_code}_{_slug(embed_model)}"
        log.info(f"[RAG] Searching collection: {coll_name}")
        hits = []
        try:
            hits = client.search(collection_name=coll_name, query_vector=q_vec, limit=20)
            log.info(f"[RAG] Collection '{coll_name}' returned {len(hits)} hits.")
            if hits:
                log.info(f"[RAG] Top hit text: {hits[0].payload.get('text') if hits[0].payload else ''}")
        except Exception as ex:
            log.warning(f"[RAG] Collection '{coll_name}' search failed: {ex}")
        if not hits:
            log.warning("[RAG] No hits found in the collection.")
            return []

        # BM25 across candidate set (not per doc)
        cand_texts=[]
        for h in hits:
            p=h.payload or {}
            cand_texts.append(p.get("text") or (p.get("metadata") or {}).get("text",""))
        tokenized = [word_tokenize(t.lower()) for t in cand_texts]
        bm25 = BM25Okapi(tokenized)
        q_tokens=word_tokenize(query.lower())
        bm_raw = bm25.get_scores(q_tokens)
        bm_min, bm_max = float(np.min(bm_raw)), float(np.max(bm_raw))
        bm_norm=[(s-bm_min)/(bm_max-bm_min+1e-9) for s in bm_raw]

        results=[]
        for i,h in enumerate(hits):
            doc=h.payload or {}; meta=doc.get("metadata",{})
            text = doc.get("text") or meta.get("text","")
            src  = doc.get("source") or meta.get("source","unknown")
            cid  = doc.get("chunk_id") or meta.get("chunk_id") or meta.get("chunk_index",0)
            tchunks = doc.get("total_chunks") or meta.get("total_chunks",1)
            d_entities = doc.get("entities") or meta.get("entities",{})
            if isinstance(d_entities,str):
                try: d_entities=json.loads(d_entities)
                except: d_entities={}
            d_phrases = doc.get("key_phrases") or meta.get("key_phrases",[])
            if isinstance(d_phrases,str):
                try: d_phrases=json.loads(d_phrases)
                except: d_phrases=[]
            d_sent = doc.get("sentiment") or meta.get("sentiment",{})

            v=h.score
            e=_ent_score(q_entities,d_entities)
            p=_phrase_score(q_phrases,d_phrases)
            s=_sent_score(q_sent,d_sent)
            b=bm_norm[i]
            final=0.55*v+0.20*b+0.15*e+0.10*p+0.00*s

            matched={}
            if isinstance(d_entities,dict) and isinstance(q_entities,dict):
                for cat,ents in d_entities.items():
                    if cat in q_entities:
                        qset=set(x.lower() for x in q_entities[cat])
                        matched[cat]=[x for x in ents if x.lower() in qset]

            results.append({"text":text,"score":float(final),"vector_score":float(v),"entity_score":float(e),
                            "source":src,"chunk_id":int(cid),"total_chunks":int(tchunks),
                            "language":language,"matched_entities":matched})
        results.sort(key=lambda x:x["score"], reverse=True)
        return results[:10]
    except Exception as e:
        log.error(f"search_documents err: {e}")
        return []

def _ent_score(q:Dict[str,List[str]], d:Dict[str,List[str]])->float:
    try:
        if not q or not d: return 0.0
        if isinstance(d,str):
            try: d=json.loads(d)
            except: return 0.0
        matches=0; total=sum(len(v) for v in q.values()) or 1
        for cat,ents in q.items():
            dset=set(x.lower() for x in d.get(cat,[]))
            for e in ents:
                if e.lower() in dset: matches+=1
        return matches/total
    except: return 0.0

def _phrase_score(qp:List[str], dp:List[str])->float:
    try:
        if not qp or not dp: return 0.0
        if isinstance(dp,str):
            try: dp=json.loads(dp)
            except: return 0.0
        qset=set(x.lower() for x in qp); dset=set(x.lower() for x in dp)
        inter=qset & dset
        return len(inter)/(len(qset) or 1)
    except: return 0.0

def _sent_score(qs:Dict[str,float], ds:Dict[str,float])->float:
    try:
        if not qs or not ds or not isinstance(ds,dict): return 0.0
        return 0.4*(1-abs(qs.get("positive",0)-ds.get("positive",0))) + \
               0.2*(1-abs(qs.get("neutral",0)-ds.get("neutral",0))) + \
               0.4*(1-abs(qs.get("negative",0)-ds.get("negative",0)))
    except: return 0.0

def generate_response(query:str, results:List[Dict[str,Any]], gen_model:str=DEFAULT_GEN_MODEL)->str:
    try:
        if not results:
            return "لم أجد معلومات كافية في الوثائق المتاحة." if any('\u0600'<=c<='\u06FF' for c in query) \
                   else "I couldn't find specific information in the available documents."
        ctx=[]
        for i,r in enumerate(results[:3],1):
            ctx.append(f"Source {i} (Score: {r.get('score',0):.2f}, Document: {r.get('source','unknown')}):\n{r.get('text','')}\n")
        context="\n".join(ctx)
        is_ar = any('\u0600'<=c<='\u06FF' for c in query)
        sys_ar="أنت مساعد دقيق. أجب بجملة واحدة وبنفس لغة السؤال اعتماداً فقط على المصادر أدناه."
        sys_en="You are precise. Answer in one clear sentence, same language as the question, using only the sources below."
        resp=ollama.chat(model=gen_model, messages=[
            {"role":"system","content": sys_ar if is_ar else sys_en},
            {"role":"user","content": f"Question: {query}\n\nSources:\n{context}\n\nAnswer in one sentence using ONLY these sources."}
        ])
        return resp["message"]["content"].strip()
    except Exception as e:
        log.error(f"gen_response err: {e}")
        return "عذراً، حدث خطأ." if any('\u0600'<=c<='\u06FF' for c in query) else "Sorry, something went wrong."