
from pathlib import Path
import pandas as pd
import re
import difflib
import unicodedata

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

FILES = [
    {
        "language": "Arabic",
        "txt": RAW / "pdpl_ar.txt",
        "csv": PROCESSED / "pdpl_ar_articles.csv",
        "article_number": "1",
        "heading_pattern": r"المادة\s+الأولى\s*:",
    },
    {
        "language": "English",
        "txt": RAW / "pdpl_en.txt",
        "csv": PROCESSED / "pdpl_en_articles.csv",
        "article_number": "1",
        "heading_pattern": r"(?im)^\s*Article\s+1\b",
    },
]


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "").replace("\u200f", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_source_article(text, pattern):
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    start = matches[0].start()
    # Find the next article heading after the first one.
    next_heading = re.search(
        r"(?im)^\s*Article\s+2\b",
        text[matches[0].end():],
    ) if "Article" in pattern else re.search(
        r"المادة\s+الثانية\s*:",
        text[matches[0].end():],
    )

    if next_heading:
        end = matches[0].end() + next_heading.start()
        return text[start:end].strip()

    return text[start:].strip()


def main():
    for item in FILES:
        print("\n" + "=" * 85)
        print(f"LANGUAGE: {item['language']}")
        print("=" * 85)

        if not item["txt"].exists() or not item["csv"].exists():
            print("ERROR: TXT or CSV file not found.")
            print("TXT:", item["txt"], item["txt"].exists())
            print("CSV:", item["csv"], item["csv"].exists())
            continue

        source_text = item["txt"].read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        df = pd.read_csv(
            item["csv"],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

        rows = df[
            df["article_number"].str.strip()
            == item["article_number"]
        ]

        if len(rows) != 1:
            print("ERROR: Expected exactly one CSV row for article 1.")
            print("Rows found:", len(rows))
            continue

        csv_text = rows.iloc[0]["article_text"]
        source_article = get_source_article(
            source_text, item["heading_pattern"]
        )

        if source_article is None:
            print("ERROR: Could not locate article 1 in TXT.")
            continue

        print("\n--- TXT SOURCE: ARTICLE 1 ---\n")
        print(source_article)

        print("\n--- CSV TEXT: ARTICLE 1 ---\n")
        print(csv_text)

        source_normalized = normalize(source_article)
        csv_normalized = normalize(csv_text)

        print("\n--- COMPARISON SUMMARY ---")
        print("TXT normalized length:", len(source_normalized))
        print("CSV normalized length:", len(csv_normalized))
        print("Exact normalized match:",
              source_normalized == csv_normalized)

        if source_normalized != csv_normalized:
            print("\n--- DIFFERENCES (first 100 lines) ---")
            diff = difflib.unified_diff(
                source_normalized.split(),
                csv_normalized.split(),
                fromfile="TXT",
                tofile="CSV",
                lineterm="",
            )
            for i, line in enumerate(diff):
                if i >= 100:
                    print("... Output truncated.")
                    break
                print(line)

    print("\nRead-only check completed. No files were modified.")


if __name__ == "__main__":
    main()
