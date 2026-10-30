"""Cross-encoder reranking of the candidates produced by hybrid retrieval."""

import os

from sentence_transformers import CrossEncoder

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def passage_text(doc) -> str:
    """Title + text of an article, as seen by the cross-encoder."""
    title = (doc.get("article_title") or "").strip()
    text = doc.get("article_text") or ""
    return f"{title}\n{text}" if title else text


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL, max_length=512):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, question, candidates, top_n=5):
        """Score every (question, article) pair and return the best top_n."""
        if not candidates:
            return []

        pairs = [(question, passage_text(doc)) for doc in candidates]
        scores = self.model.predict(
            pairs, batch_size=8, show_progress_bar=False
        )

        reranked = []
        for doc, score in zip(candidates, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda d: d["rerank_score"], reverse=True)

        for rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rank

        return reranked[:top_n]
