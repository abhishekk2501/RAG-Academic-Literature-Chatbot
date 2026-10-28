# Free PDF RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot that allows a user to upload a PDF and ask questions about its contents.

## Technologies

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- Ollama
- Qwen 2.5 3B

No OpenAI API key is required.

## 1. Create the environment

Open Command Prompt in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install Python packages

```bash
pip install -r requirements.txt
```

## 3. Install Ollama

Install Ollama from its official website:

https://ollama.com/

After installation, open Command Prompt and run:

```bash
ollama pull qwen2.5:3b
```

Then test it:

```bash
ollama run qwen2.5:3b
```

Type a question. Press Ctrl+C when finished.

Ollama normally runs at:

```text
http://localhost:11434
```

## 4. Start the chatbot

In the project folder:

```bash
streamlit run app.py
```

The browser should open the Streamlit application.

## 5. How the RAG pipeline works

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
ChromaDB Vector Database
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Top Relevant PDF Chunks
    ↓
Ollama + Qwen 2.5
    ↓
Answer
```

## Important

This project works best with PDFs containing selectable text. Scanned/image-only PDFs require OCR, which is not included in this basic version.
