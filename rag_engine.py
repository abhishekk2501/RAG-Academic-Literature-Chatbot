import os
import requests
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.Client()
        self.collection = None

        self.ollama_url = os.getenv(
            "OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    def _chunk_text(self, text, chunk_size=900, overlap=150):
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap

        return chunks

    def index_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)

        documents = []
        metadatas = []
        ids = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())

            if not text:
                continue

            chunks = self._chunk_text(text)

            for chunk_number, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append(
                    {
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )
                ids.append(f"page_{page_number}_chunk_{chunk_number}")

        if not documents:
            raise ValueError(
                "No readable text was found. The PDF may be scanned/image-only."
            )

        # Recreate collection for the newly uploaded PDF.
        try:
            self.client.delete_collection("pdf_documents")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return len(documents)

    def retrieve(self, question, top_k=5):
        if self.collection is None:
            raise ValueError("Please upload a PDF first.")

        query_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        sources = [
            {"text": doc, "page": meta.get("page", "?")}
            for doc, meta in zip(documents, metadatas)
        ]

        return sources

    def ask(self, question):
        sources = self.retrieve(question, top_k=5)

        context = "\n\n".join(
            f"[Page {source['page']}]\n{source['text']}"
            for source in sources
        )

        prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information in the provided PDF context.

Rules:
- Do not invent facts.
- If the answer is not present in the context, say:
  "I could not find this information in the uploaded PDF."
- Keep the answer clear and concise.
- When useful, mention the relevant page number.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=180,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError("Ollama returned an empty response.")

        return answer, sources
