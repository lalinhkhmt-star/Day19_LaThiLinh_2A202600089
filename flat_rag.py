"""
Flat RAG - Hệ thống RAG truyền thống dùng TF-IDF (ChromaDB/Faiss simulation)
"""

import math
import re
import time
from collections import Counter
from corpus import TECH_CORPUS


def tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def compute_tf(tokens: list) -> dict:
    count = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in count.items()}


def compute_idf(docs: list) -> dict:
    N = len(docs)
    df = {}
    for doc in docs:
        for word in set(tokenize(doc)):
            df[word] = df.get(word, 0) + 1
    return {w: math.log(N / (freq + 1)) + 1 for w, freq in df.items()}


def tfidf_vector(tokens: list, idf: dict) -> dict:
    tf = compute_tf(tokens)
    return {w: tf[w] * idf.get(w, 0) for w in tf}


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    keys = set(vec1) | set(vec2)
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class FlatRAG:
    """TF-IDF based Flat RAG (simulates ChromaDB/Faiss retrieval)."""

    def __init__(self, corpus: list = None):
        self.corpus = corpus or TECH_CORPUS
        self.idf = compute_idf(self.corpus)
        self.doc_vectors = [
            tfidf_vector(tokenize(doc), self.idf) for doc in self.corpus
        ]
        print(f"✅ FlatRAG indexed {len(self.corpus)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        q_vec = tfidf_vector(tokenize(query), self.idf)
        scores = [cosine_similarity(q_vec, dv) for dv in self.doc_vectors]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            results.append({"doc": self.corpus[idx], "score": round(score, 4)})
        return results

    def answer(self, query: str, top_k: int = 3) -> dict:
        start = time.time()
        retrieved = self.retrieve(query, top_k)
        context = "\n".join([r["doc"] for r in retrieved])
        elapsed = time.time() - start

        # Simulate LLM answer from context
        answer_text = self._simulate_answer(query, context)
        return {
            "query": query,
            "context": context,
            "answer": answer_text,
            "retrieved_docs": retrieved,
            "latency_ms": round(elapsed * 1000, 2),
            "method": "FlatRAG (TF-IDF)",
        }

    def _simulate_answer(self, query: str, context: str) -> str:
        """Simulate answer generation from retrieved context."""
        q_lower = query.lower()
        sentences = context.split(".")
        relevant = [s.strip() for s in sentences if s.strip()]

        # Simple keyword matching to find most relevant sentence
        best = ""
        best_score = -1
        q_words = set(tokenize(query))
        for sent in relevant:
            overlap = len(q_words & set(tokenize(sent)))
            if overlap > best_score:
                best_score = overlap
                best = sent

        if best:
            return f"Dựa trên tài liệu: {best}."
        return "Không tìm thấy thông tin liên quan trong tài liệu."


if __name__ == "__main__":
    rag = FlatRAG()
    result = rag.answer("OpenAI được thành lập năm nào?")
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Latency: {result['latency_ms']} ms")
