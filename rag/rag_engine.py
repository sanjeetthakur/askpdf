import os

import numpy as np
import requests

from rag.chunking import chunk_text
from rag.document_store import DocumentStore
from rag.ollama_client import OllamaClient
from rag.vectorizer import LocalTfidfVectorizer


class RagEngine:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.retrieval_model_name = os.getenv("RETRIEVAL_MODEL", "local-tfidf")
        self.llm = OllamaClient()
        self.ollama_model = self.llm.model

    def index_document(self, doc_id: str, filename: str, text: str, page_count: int) -> dict[str, str | int]:
        chunks = chunk_text(text)
        vectorizer = LocalTfidfVectorizer()
        vectors = vectorizer.fit_transform(chunks)

        self.store.save(
            doc_id,
            {
                "doc_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "chunks": chunks,
                "vectors": vectors,
                "vectorizer": vectorizer,
                "retrieval_model": self.retrieval_model_name,
            },
        )

        return {
            "chunk_count": len(chunks),
            "preview": chunks[0][:450],
        }

    def answer_question(self, doc_id: str, question: str, top_k: int = 5) -> dict[str, object]:
        index = self.store.load(doc_id)
        chunks: list[str] = index["chunks"]
        vectors = np.array(index["vectors"], dtype=np.float32)
        vectorizer: LocalTfidfVectorizer = index["vectorizer"]

        query_vector = vectorizer.transform([question])[0]
        scores = cosine_similarity(query_vector, vectors)
        best_indices = scores.argsort()[-top_k:][::-1]

        sources = [
            {
                "rank": rank + 1,
                "score": round(float(scores[index]), 4),
                "text": chunks[index],
            }
            for rank, index in enumerate(best_indices)
        ]
        context = "\n\n".join(f"Source {item['rank']}:\n{item['text']}" for item in sources)
        prompt = build_prompt(question, context)

        try:
            answer = self.llm.generate(prompt)
            mode = "local_llm"
        except requests.RequestException:
            answer = fallback_answer(question, sources)
            mode = "extractive_fallback"

        return {
            "answer": answer,
            "mode": mode,
            "model": self.ollama_model if mode == "local_llm" else "retrieval-only fallback",
            "sources": sources,
        }

def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ query_embedding


def build_prompt(question: str, context: str) -> str:
    return f"""You are a local PDF question-answering assistant.
Answer only from the provided PDF context.
If the answer is not in the context, say that the PDF does not contain enough information.
Keep the answer clear, concise, and useful.

PDF context:
{context}

Question: {question}

Answer:"""


def fallback_answer(question: str, sources: list[dict[str, object]]) -> str:
    best = "\n\n".join(f"- {source['text'][:550].strip()}" for source in sources[:3])
    return (
        "I found the most relevant PDF sections, but the local Ollama model is not reachable. "
        "Start Ollama and pull the configured model for generated answers.\n\n"
        f"Question: {question}\n\nRelevant evidence:\n{best}"
    )
