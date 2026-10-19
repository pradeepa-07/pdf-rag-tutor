import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from rag import index_pdf, ask_stream

app = FastAPI()
os.makedirs("uploads", exist_ok=True)


class AskBody(BaseModel):
    doc_id: str
    question: str
    history: list = []


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    path = os.path.join("uploads", file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc_id, n_chunks = index_pdf(path)
    return {"doc_id": doc_id, "chunks": n_chunks}


def safe_stream(doc_id, question, history):
    """Show errors in the chat instead of silently returning nothing."""
    got_text = False
    try:
        for token in ask_stream(doc_id, question, history):
            got_text = True
            yield token
        if not got_text:
            yield "[The model returned an empty answer.]"
    except Exception as e:
        print("ASK ERROR:", repr(e))
        yield f"\n[Error: {e}]"


@app.post("/ask")
def ask(body: AskBody):
    return StreamingResponse(
        safe_stream(body.doc_id, body.question, body.history),
        media_type="text/plain",
    )