
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

files = [
    ("pdpl_ar_articles.csv", "Arabic"),
    ("pdpl_en_articles.csv", "English"),
]

for filename, language in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    article = df[
        df["article_number"].str.strip() == "32"
    ]

    if article.empty:
        print("Article 32 was not found.")
        continue

    for _, row in article.iterrows():
        print(f"Language: {language}")
        print(f"Article number: {row['article_number']}")
        print(f"Article title: {row['article_title']!r}")
        print(f"Text length: {len(row['article_text'])}")
        print("\nArticle text:")
        print(row["article_text"])

    print("\nRead-only inspection: no files modified.")
