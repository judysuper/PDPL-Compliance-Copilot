
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

index = faiss.read_index(str(INDEX_DIR / "pdpl_faiss.index"))
metadata = pd.read_csv(INDEX_DIR / "pdpl_metadata.csv")

model = SentenceTransformer("intfloat/multilingual-e5-base")

question = "ما حقوق صاحب البيانات الشخصية؟"
query_embedding = model.encode(
    ["query: " + question],
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype("float32")

scores, indices = index.search(query_embedding, 5)

print(f"Question: {question}")
print(f"Vectors in index: {index.ntotal}")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]), start=1
):
    if idx < 0:
        continue

    article = metadata.iloc[idx]

    print(f"\n--- Result {rank} | Score: {score:.4f} ---")
    print("Source:", article["source_csv"])
    print("Article:", article["article_number"])
    print("Title:", article["article_title"])
    print("Text:", str(article["article_text"])[:400])
