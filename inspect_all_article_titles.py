
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# TXT source and corresponding CSV for each legal document
DOCUMENTS = [
    {
        "name": "PDPL Arabic",
        "txt": "pdpl_ar.txt",
        "csv": "pdpl_ar_articles.csv",
        "language": "ar",
    },
    {
        "name": "PDPL English",
        "txt": "pdpl_en.txt",
        "csv": "pdpl_en_articles.csv",
        "language": "en",
    },
    {
        "name": "Implementing Regulations Arabic",
        "txt": "implement_ar.txt",
        "csv": "implementing_ar_articles.csv",
        "language": "ar",
    },
    {
        "name": "Implementing Regulations English",
        "txt": "implementing_en.txt",
        "csv": "implementing_en_articles.csv",
        "language": "en",
    },
    {
        "name": "Transfer Regulations Arabic",
        "txt": "transfer_ar.txt",
        "csv": "transfer_ar_articles.csv",
        "language": "ar",
    },
    {
        "name": "Transfer Regulations English",
        "txt": "transfer_en.txt",
        "csv": "transfer_en_articles.csv",
        "language": "en",
    },
]


def remove_page_markers(text):
    """Remove page markers for inspection only."""
    text = re.sub(
        r"={2,}\s*PAGE\s*\d+\s*(?:/\s*\d+)?\s*={2,}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"={2,}\s*صفحة\s*\d+\s*={2,}",
        " ",
        text,
    )
    return text


def detect_articles(text, language):
    """Detect article boundaries without editing source files."""
    if language == "en":
        pattern = re.compile(
            r"(?im)^[ \t]*Article[ \t]+(\d+)\b[ \t]*(.*)$"
        )
    else:
        pattern = re.compile(
            r"(?m)^[ \t]*المادة[ \t]+"
            r"([^\s:：]+)[ \t]*[:：]?[ \t]*(.*)$"
        )

    matches = list(pattern.finditer(text))
    articles = []

    for i, match in enumerate(matches):
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        article_block = text[match.start():end]
        article_block = remove_page_markers(article_block)

        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in article_block.splitlines()
        ]
        lines = [line for line in lines if line]

        heading_line = (
            re.sub(r"\s+", " ", match.group(2)).strip()
        )

        # Remove the article number line from the body preview.
        body_lines = lines[1:] if len(lines) > 1 else []

        articles.append({
            "number": match.group(1),
            "heading_remainder": heading_line,
            "body_preview": " ".join(body_lines)[:300],
            "block_preview": " ".join(lines)[:350],
        })

    return articles


def inspect_document(document):
    name = document["name"]
    txt_path = RAW_DIR / document["txt"]
    csv_path = PROCESSED_DIR / document["csv"]
    language = document["language"]

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    if not txt_path.exists():
        print(f"TXT NOT FOUND: {txt_path}")
        return

    if not csv_path.exists():
        print(f"CSV NOT FOUND: {csv_path}")
        return

    text = txt_path.read_text(encoding="utf-8-sig")
    articles = detect_articles(text, language)

    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    print(f"TXT: {txt_path.name}")
    print(f"CSV: {csv_path.name}")
    print(f"TXT articles detected: {len(articles)}")
    print(f"CSV rows: {len(df)}")

    if "article_title" in df.columns:
        titles = df["article_title"].astype(str).str.strip()
        filled = (titles != "").sum()
        print(f"Non-empty CSV titles: {filled}")
        print(f"Empty CSV titles: {len(df) - filled}")
    else:
        print("WARNING: article_title column is missing.")

    print("\nFirst 5 detected articles:")
    for article in articles[:5]:
        print(f"\nArticle number: {article['number']}")
        print(
            "Text on heading line: "
            + (article["heading_remainder"] or "[EMPTY]")
        )
        print(
            "Following text preview: "
            + (article["body_preview"] or "[EMPTY]")
        )

    print("\nInspection only: no files were modified.")


if __name__ == "__main__":
    for document in DOCUMENTS:
        inspect_document(document)

    print("\n" + "=" * 90)
    print("ALL SIX DOCUMENTS INSPECTED")
    print("Read-only check completed. No files were modified.")
    print("=" * 90)
