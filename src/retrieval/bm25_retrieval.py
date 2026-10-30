
import csv
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_FILES = [
    "pdpl_ar_articles.csv",
    "pdpl_en_articles.csv",
    "implementing_ar_articles.csv",
    "implementing_en_articles.csv",
    "transfer_ar_articles.csv",
    "transfer_en_articles.csv",
]


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize Arabic or English text."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.lower()

    # Remove Arabic diacritics and tatweel
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)

    # Normalize common Arabic letter variants
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")

    # Extract Arabic words, English words, and numbers
    return re.findall(r"[\u0600-\u06FF]+|[a-z0-9]+", text)


def load_legal_documents() -> list[dict]:
    """Load legal articles and metadata from the six CSV files."""
    documents = []

    for filename in EXPECTED_FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required CSV file was not found: {file_path}"
            )

        # Infer source and language from the filename
        source = filename.split("_")[0]
        language = "ar" if "_ar_" in filename else "en"

        with file_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            required_columns = {
                "document_id",
                "article_number",
                "article_text",
            }

            if not reader.fieldnames or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Missing required columns in {filename}"
                )

            for row in reader:
                article_text = (row.get("article_text") or "").strip()
                article_title = (row.get("article_title") or "").strip()

                if not article_text:
                    continue

                documents.append({
                    "document_id": row.get("document_id", ""),
                    "language": row.get("language") or language,
                    "article_number": row.get("article_number", ""),
                    "article_title": article_title,
                    "article_text": article_text,
                    "source": source,
                    "source_file": filename,
                    "source_line_start": row.get(
                        "source_line_start", ""
                    ),
                })

    return documents


def build_indexes():
    """Build separate BM25 indexes for Arabic and English."""
    documents = load_legal_documents()

    documents_by_language = {
        "ar": [],
        "en": [],
    }

    for document in documents:
        language = document["language"].strip().lower()

        if language in documents_by_language:
            documents_by_language[language].append(document)

    indexes = {}

    for language, language_documents in documents_by_language.items():
        if not language_documents:
            continue

        tokenized_documents = [
            tokenize(
                f"{doc['article_title']} {doc['article_text']}"
            )
            for doc in language_documents
        ]

        indexes[language] = {
            "bm25": BM25Okapi(tokenized_documents),
            "documents": language_documents,
        }

    return indexes


def detect_language(text: str) -> str:
    """Choose Arabic if the query contains Arabic characters."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"

    return "en"


def search(
    query: str,
    top_k: int = 5,
    language: str | None = None,
    indexes: dict | None = None,
) -> list[dict]:
    """Return the highest-scoring legal articles for a query."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if indexes is None:
        indexes = build_indexes()

    language = language or detect_language(query)

    if language not in indexes:
        raise ValueError(
            f"No BM25 index available for language: {language}"
        )

    index = indexes[language]
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = index["bm25"].get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for rank, i in enumerate(ranked_indices, start=1):
        document = index["documents"][i]

        results.append({
            "rank": rank,
            "score": float(scores[i]),
            **document,
        })

    return results


if __name__ == "__main__":
    print("Loading legal articles and building BM25 indexes...")

    bm25_indexes = build_indexes()

    for language, index in bm25_indexes.items():
        print(
            f"{language}: "
            f"{len(index['documents'])} legal articles indexed"
        )

    test_query = "نقل البيانات الشخصية خارج المملكة"
    results = search(
        test_query,
        top_k=5,
        language="ar",
        indexes=bm25_indexes,
    )

    print(f"\nTest query: {test_query}")
    print("\nTop results:")

    for result in results:
        print(
            f"\nRank: {result['rank']}"
            f" | Score: {result['score']:.4f}"
            f" | Source: {result['source']}"
            f" | Article: {result['article_number']}"
        )
        print(f"Title: {result['article_title'] or '(No title)'}")
        print(f"Text: {result['article_text'][:250]}...")
