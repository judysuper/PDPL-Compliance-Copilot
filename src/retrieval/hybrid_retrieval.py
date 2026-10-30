"""Hybrid retrieval: BM25 + dense (FAISS) results fused with Reciprocal Rank Fusion."""

import re
from collections import defaultdict
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval.bm25_retrieval import (
    build_indexes,
    detect_language,
    search as bm25_search,
)
from retrieval.dense_retrieval import INDEX_DIR, MODEL_NAME

RRF_K = 60


def normalize_number(value) -> str:
    """'5.0' -> '5' (pandas may turn integer columns into floats)."""
    return re.sub(r"\.0+$", "", str(value).strip())


def make_article_id(csv_name, document_id, article_number) -> str:
    """Stable id shared by BM25 and dense results for the same article."""
    return f"{csv_name}::{document_id}::{normalize_number(article_number)}"


def bm25_candidates(question, indexes, top_k=10, language=None):
    results = bm25_search(
        question, top_k=top_k, language=language, indexes=indexes
    )
    for r in results:
        r["article_id"] = make_article_id(
            r["source_file"], r["document_id"], r["article_number"]
        )
    return results


class DenseSearcher:
    """Loads the FAISS index built by dense_retrieval.py and answers queries."""

    def __init__(self, index_dir=INDEX_DIR, model_name=MODEL_NAME):
        index_path = Path(index_dir) / "pdpl_faiss.index"
        meta_path = Path(index_dir) / "pdpl_metadata.csv"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "FAISS index or metadata not found. "
                "Run src/retrieval/dense_retrieval.py first."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_csv(
            meta_path, encoding="utf-8-sig", dtype=str, keep_default_na=False
        )

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) != "
                f"index vectors ({self.index.ntotal}). Rebuild the index."
            )

        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=10, language=None):
        # The corpus is small and the index is flat, so when filtering by
        # language we simply scan everything and keep the first top_k matches.
        fetch_k = self.index.ntotal if language else top_k

        vector = self.model.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, ids = self.index.search(vector, fetch_k)

        results = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue

            row = self.metadata.iloc[int(i)]
            csv_name = row["source_csv"]
            row_lang = (row.get("language") or "").strip().lower() or (
                "ar" if "_ar_" in csv_name else "en"
            )

            if language and row_lang != language:
                continue

            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "document_id": row["document_id"],
                "language": row_lang,
                "article_number": normalize_number(row["article_number"]),
                "article_title": row["article_title"].strip(),
                "article_text": row["article_text"],
                "source": csv_name.split("_")[0],
                "source_file": csv_name,
                "article_id": make_article_id(
                    csv_name, row["document_id"], row["article_number"]
                ),
            })

            if len(results) >= top_k:
                break

        return results


def rrf_fusion(result_lists, names=None, k=RRF_K):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    names = names or [f"list{i}" for i in range(len(result_lists))]

    scores = defaultdict(float)
    ranks = defaultdict(dict)
    docs = {}

    for name, results in zip(names, result_lists):
        for rank, doc in enumerate(results, start=1):
            article_id = doc["article_id"]
            scores[article_id] += 1.0 / (k + rank)
            ranks[article_id][name] = rank
            docs.setdefault(article_id, doc)

    fused = []
    for article_id in sorted(scores, key=scores.get, reverse=True):
        item = {
            key: value
            for key, value in docs[article_id].items()
            if key not in ("rank", "score")
        }
        item["rrf_score"] = scores[article_id]
        for name in names:
            item[f"{name}_rank"] = ranks[article_id].get(name)
        fused.append(item)

    return fused


def hybrid_search(
    question,
    bm25_indexes,
    dense,
    top_k=10,
    fused_k=20,
    language=None,
    cross_lingual=False,
    rrf_k=RRF_K,
):
    """Run BM25 and dense search, then fuse them with RRF."""
    language = language or detect_language(question)

    bm25_results = bm25_candidates(
        question, bm25_indexes, top_k=top_k, language=language
    )
    dense_results = dense.search(
        question,
        top_k=top_k,
        language=None if cross_lingual else language,
    )

    fused = rrf_fusion(
        [bm25_results, dense_results], names=["bm25", "dense"], k=rrf_k
    )[:fused_k]

    return {
        "language": language,
        "bm25": bm25_results,
        "dense": dense_results,
        "fused": fused,
    }
