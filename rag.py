import uuid
import fitz  # PyMuPDF
import chromadb
import ollama

LLM_MODEL = "llama3.2:3b"  # no thinking mode, so no leaked reasoning
EMBED_MODEL = "nomic-embed-text"

CHUNK_WORDS = 350      # about 500 tokens
OVERLAP_WORDS = 50     # about 15% overlap
TOP_K = 5



client = chromadb.PersistentClient(path="./chroma_db")

TUTOR_PROMPT = """You are a friendly tutor and mentor helping a student with a PDF.

Use the document context below as your main evidence, and mention page
numbers like (page 4) when you use it. Never invent facts about the document;
if a fact is not in the context, say so.

If the question asks for explanation, opinion or analysis (for example
"will these skills be replaced by AI?" or "what should I learn next?"),
reason from the facts in the document PLUS your general knowledge, and give
a clear, balanced answer. Make it obvious which points come from the document
and which are your general knowledge.

Explain in simple words, with an analogy or small example when it helps.
Start directly with the answer to the student's question.

Context:
{context}
"""


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def chunk_pages(pages):
    """Split each page into overlapping word windows, keeping the page number."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page_no, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if len(piece.split()) > 20:  # skip tiny scraps
                chunks.append({"text": piece, "page": page_no})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


def embed(texts, prefix):
    out = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return out["embeddings"]


def index_pdf(pdf_path):
    """Read, chunk, embed and store a PDF. Returns a doc_id."""
    chunks = chunk_pages(extract_pages(pdf_path))
    if not chunks:
        raise ValueError("No text found. Scanned PDFs need OCR first.")

    doc_id = "doc_" + uuid.uuid4().hex[:8]
    col = client.create_collection(doc_id, metadata={"hnsw:space": "cosine"})

    batch = 32
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(
            ids=[f"{doc_id}_{i + j}" for j in range(len(part))],
            documents=[c["text"] for c in part],
            embeddings=embed([c["text"] for c in part], "search_document: "),
            metadatas=[{"page": c["page"]} for c in part],
        )
    return doc_id, len(chunks)


def strip_think(stream):
    """Hide any <think>...</think> reasoning that Qwen3 may print."""
    buf, in_think, started = "", False, False
    for part in stream:
        buf += part["message"]["content"]
        while True:
            if in_think:
                end = buf.find("</think>")
                if end == -1:
                    buf = buf[-8:]  # keep a tail in case the tag is split
                    break
                buf = buf[end + 8:]
                in_think = False
            else:
                start = buf.find("<think>")
                if start == -1:
                    if len(buf) > 7:
                        out, buf = buf[:-7], buf[-7:]
                        if not started:
                            out = out.lstrip()
                        if out:
                            started = True
                            yield out
                    break
                if start > 0:
                    started = True
                    yield buf[:start]
                buf = buf[start + 7:]
                in_think = True
    if not in_think and buf:
        yield buf if started else buf.lstrip()


def ask_stream(doc_id, question, history=None):
    """Yield the tutor's answer token by token."""
    col = client.get_collection(doc_id)
    q_emb = embed([question], "search_query: ")
    res = col.query(query_embeddings=q_emb, n_results=TOP_K)

    context = "\n\n".join(
        f"[page {m['page']}] {d}"
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    )

    messages = [{"role": "system", "content": TUTOR_PROMPT.format(context=context)}]
    messages += (history or [])[-6:]  # last few turns for follow-up doubts
    messages.append({"role": "user", "content": question})

    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)

    yield from strip_think(stream)