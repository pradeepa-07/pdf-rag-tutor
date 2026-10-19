# PDF RAG Tutor

A small PDF tutoring app that lets you upload a document, index it for retrieval, and ask questions about the PDF in natural language. The app uses a RAG pipeline with ChromaDB and Ollama to answer questions grounded in the uploaded document.

## Features

- Upload a PDF from the browser
- Extract and chunk PDF pages
- Embed and store document chunks in ChromaDB
- Retrieve relevant passages for a user question
- Generate a tutor-style answer using Ollama
- Keep a short conversation history for follow-up questions

## Project Structure

- `app.py` – FastAPI server and API endpoints
- `rag.py` – PDF parsing, chunking, vector indexing, and streaming answers
- `index.html` – Simple browser UI for uploading PDFs and asking questions
- `requirement.txt` – Python dependencies
- `uploads/` – Uploaded PDF files (created automatically)
- `chroma_db/` – ChromaDB vector database storage (created automatically)

## Tech Stack

- Python
- FastAPI
- ChromaDB
- PyMuPDF (`fitz`)
- Ollama
- HTML/JavaScript frontend

## Prerequisites

- Python 3.10+
- Ollama installed and running locally
- Access to the required models:
  - `qwen3:4b`
  - `nomic-embed-text`

## Setup

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirement.txt
   ```

4. Start Ollama and pull the required models:

   ```bash
   ollama pull qwen3:4b
   ollama pull nomic-embed-text
   ```

5. Run the FastAPI app:

   ```bash
   uvicorn app:app --reload
   ```

6. Open the browser at:

   ```text
   http://localhost:8000
   ```

## Usage

1. Upload a PDF using the file picker.
2. Wait for the document to be indexed.
3. Ask a question about the uploaded PDF.
4. The app retrieves relevant document chunks and streams a tutor-style answer.

## Notes

- This app is best suited for PDFs with searchable text.
- Scanned PDFs may require OCR before indexing.
- The app stores the vector database locally in the `chroma_db` folder.

## Example Workflow

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
ollama pull qwen3:4b
ollama pull nomic-embed-text
uvicorn app:app --reload
```

Then visit `http://localhost:8000` and upload a PDF.
