<h1>Development of a PDF-Based Question-Answering Chatbot Using LangChain</h1>

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
This project creates a PDF chatbot using Google Gemini. It extracts and chunks PDF text, generates embeddings, and retrieves relevant content to answer user questions quickly and accurately.

### DESIGN STEPS:

#### STEP 1:
Read the PDF and extract text from each page.Split the text into manageable chunks with overlap for better retrieval.

#### STEP 2:
Generate embeddings for each chunk using Gemini embeddings.Cache embeddings to speed up repeated runs.

#### STEP 3:
Take a user question, find the most relevant PDF chunks using cosine similarity.Feed the retrieved chunks to Gemini LLM to generate an answer based only on the PDF content.

### PROGRAM:
```
import os, json
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai


GOOGLE_API_KEY = "AIzaSyChL3CmyEpJnCIkuOqSEmOWjv5XHdySSIE"
PDF_PATH       = "/content/lora.pdf"
EMBED_MODEL    = "text-embedding-004"
MODEL_CANDIDATES = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"]

CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K          = 5

CACHE_DIR      = "/content/.rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
EMBED_CACHE    = os.path.join(CACHE_DIR, "embeds.npy")
CHUNK_CACHE    = os.path.join(CACHE_DIR, "chunks.json")

genai.configure(api_key=GOOGLE_API_KEY)

def extract_pdf_text_with_pages(pdf_path):
    reader = PdfReader(pdf_path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append({"page": i, "text": txt})
    return out

def chunk_page_text(page, text, size=1000, overlap=200):
    text = " ".join((text or "").split())
    chunks, start, L = [], 0, len(text)
    if L == 0: return chunks
    while start < L:
        end = min(start + size, L)
        chunks.append({"page": page, "text": text[start:end]})
        if end == L: break
        start = max(0, end - overlap)
    return chunks

def chunk_pdf(pages, size=1000, overlap=200):
    out = []
    for item in pages:
        if item["text"].strip():
            out.extend(chunk_page_text(item["page"], item["text"], size, overlap))
    return out

def _to_2d(v):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:     return v.reshape(1, -1)
    if v.ndim >= 3:     return v.reshape(-1, v.shape[-1])
    return v

def embed_single(text):
    # Increased timeout to 120 seconds
    res = genai.embed_content(model=EMBED_MODEL, content=text, request_options={'timeout': 120})
    emb = None
    if isinstance(res, dict):
        emb = res.get("embedding") or (res.get("embeddings") or [None])[0]
    else:
        emb = getattr(res, "embedding", None) or getattr(res, "embeddings", None)

    if isinstance(emb, dict) and "values" in emb:
        vec = np.array(emb["values"], dtype=np.float32)
    elif isinstance(emb, list):
        if len(emb) == 1 and isinstance(emb[0], (list, tuple, np.ndarray)):
            vec = np.array(emb[0], dtype=np.float32)
        else:
            vec = np.array(emb, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected embedding format: {type(emb)}")
    return _to_2d(vec)

def embed_texts(texts):
    if isinstance(texts, str): texts = [texts]
    arrs = [embed_single(t) for t in texts]
    vecs = np.vstack(arrs) if arrs else np.zeros((0, 768), dtype=np.float32)
    print(f"[DEBUG] embed_texts -> {len(texts)} inputs, shape {vecs.shape}")
    return vecs

def build_or_load_index(chunks):
    if os.path.exists(EMBED_CACHE) and os.path.exists(CHUNK_CACHE):
        try:
            with open(CHUNK_CACHE, "r") as f:
                saved_chunks = json.load(f)
            if len(saved_chunks) == len(chunks):
                vecs = np.load(EMBED_CACHE)
                if vecs.shape[0] == len(chunks):
                    print("[INFO] Loaded embeddings from cache")
                    return vecs, saved_chunks
        except Exception:
            pass

    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    np.save(EMBED_CACHE, vecs)
    with open(CHUNK_CACHE, "w") as f:
        json.dump(chunks, f)
    return vecs, chunks

def cosine_similarity(a, b):
    a = _to_2d(a); b = _to_2d(b)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

def retrieve(query, index_vecs, chunks, top_k=5):
    qvec = embed_texts(query)        # (1, d)
    sims = cosine_similarity(qvec, index_vecs)[0]
    k = min(top_k, len(chunks))
    top_idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in top_idx]
def generate_answer(prompt):
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(name)
            resp  = model.generate_content(prompt)
            txt   = getattr(resp, "text", None)
            if txt: return txt.strip(), name
        except Exception as e:
            last_err = e
            continue
    return (f"Generation unavailable for this API key."
            f"Please enable billing or use a key with generation quota.\n\nError: {last_err}"), None

def answer_with_gemini(question, retrieved):
    context = "\n---\n".join([f"[p.{c['page']}, score={s:.3f}]\n{c['text']}" for c, s in retrieved])
    prompt  = (
        "You are a precise assistant. Use ONLY the provided PDF context. "
        "If the answer isn't present, say you don't know. Cite pages like (p.X).\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )
    return generate_answer(prompt)

def main():
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Set GOOGLE_API_KEY (env or in code). Also rotate keys you’ve pasted publicly.")

    print(f"Loading PDF: {PDF_PATH}")
    pages  = extract_pdf_text_with_pages(PDF_PATH)
    chunks = chunk_pdf(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"[INFO] Extracted {len(chunks)} chunks")

    if not chunks:
        print("[WARN] No text found. If the PDF is scanned, run OCR first.")
        return

    index_vecs, chunk_refs = build_or_load_index(chunks)

    print("\nPDF Q&A Chatbot (Gemini)")
    print("Type 'quit' to exit.\n")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        hits = retrieve(q, index_vecs, chunk_refs, top_k=TOP_K)
        answer, model_used = answer_with_gemini(q, hits)
        if model_used:
            print(f"\n[Model: {model_used}]")
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    main()
```
### OUTPUT:
<img width="1280" height="266" alt="image" src="https://github.com/user-attachments/assets/4b7cacb9-ddb0-479c-a51d-73c67055e3e7" />


### RESULT:
The result is an interactive PDF chatbot that provides quick, precise answers from the PDF content without manually reading the document.
