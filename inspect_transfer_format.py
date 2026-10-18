
from pathlib import Path
import pandas as pd
import re

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FILES = {
    "Arabic": {
        "txt": RAW_DIR / "transfer_ar.txt",
        "csv": PROCESSED_DIR / "transfer_ar_articles.csv",
    },
    "English": {
        "txt": RAW_DIR / "transfer_en.txt",
        "csv": PROCESSED_DIR / "transfer_en_articles.csv",
    },
}

for language, paths in FILES.items():
    print("\n" + "=" * 70)
    print(f"{language.upper()} - SOURCE FORMAT INSPECTION")
    print("=" * 70)

    txt_path = paths["txt"]
    csv_path = paths["csv"]

    if not txt_path.exists() or not csv_path.exists():
        print("ERROR: One or both files are missing.")
        print("TXT:", txt_path)
        print("CSV:", csv_path)
        continue

    text = txt_path.read_text(encoding="utf-8-sig")
    df = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    print("\n--- First 60 lines of TXT ---")

    for line_number, line in enumerate(
        text.splitlines()[:60], start=1
    ):
        print(f"{line_number:03}: {line[:220]}")

    print("\n--- Lines that may contain article headings ---")

    keywords = (
        ["المادة", "البند", "أولاً", "أولًا", "1."]
        if language == "Arabic"
        else ["Article", "Section", "1."]
    )

    found = 0

    for line_number, line in enumerate(
        text.splitlines(), start=1
    ):
        if any(word.lower() in line.lower() for word in keywords):
            print(f"{line_number:03}: {line[:220]}")
            found += 1

            if found >= 25:
                print("... stopped after 25 matching lines")
                break

    print("\n--- First 250 characters of each CSV article ---")

    for _, row in df.iterrows():
        print(f"\nArticle number: {row['article_number']}")
        print(f"Title: {row.get('article_title', '')}")
        print(f"Text: {row['article_text'][:250]}")

print("\nInspection completed. No files were modified.")