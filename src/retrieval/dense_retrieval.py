
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

# Embedding model
MODEL_NAME = "intfloat/multilingual-e5-base"

# Legal data files
DATA_FILES = [
    "pdpl_ar_articles.csv",
    "pdpl_en_articles.csv",
    "implementing_ar_articles.csv",
    "implementing_en_articles.csv",
    "transfer_ar_articles.csv",
    "transfer_en_articles.csv",
]


def load_legal_data():
    """Load legal articles from all six CSV files."""
    frames = []

    for filename in DATA_FILES:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        df = pd.read_csv(path)

        required_columns = [
            "document_id",
            "language",
            "article_number",
            "article_title",
            "article_text",
            "source_file",
        ]

        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{filename} is missing columns: {sorted(missing)}"
            )

        df = df.dropna(subset=["article_text"]).copy()
        df["article_text"] = df["article_text"].astype(str).str.strip()
        df = df[df["article_text"] != ""]

        df["source_csv"] = filename
        frames.append(df)

        print(f"Loaded {filename}: {len(df)} articles")

    data = pd.concat(frames, ignore_index=True)
    data = data.reset_index(drop=True)

    print(f"\nTotal legal articles loaded: {len(data)}")
    return data


def build_index(data):
    """Encode legal articles and build a FAISS inner-product index."""
    model = SentenceTransformer(MODEL_NAME)

    passages = [
        "passage: " + str(text)
        for text in data["article_text"]
    ]

    embeddings = model.encode(
        passages,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "pdpl_faiss.index"
    metadata_path = INDEX_DIR / "pdpl_metadata.csv"

    faiss.write_index(index, str(index_path))
    data.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print(f"\nIndex saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Vectors in index: {index.ntotal}")


if __name__ == "__main__":
    legal_data = load_legal_data()
    build_index(legal_data)
