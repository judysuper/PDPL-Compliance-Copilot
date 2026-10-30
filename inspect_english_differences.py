
from pathlib import Path
import pandas as pd
import re
import unicodedata
import difflib

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

TXT_PATH = RAW / "pdpl_en.txt"
CSV_PATH = PROCESSED / "pdpl_en_articles.csv"


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "").replace("\u200f", "")

    # Remove page markers, including those embedded within an article.
    text = re.sub(
        r"={2,}\s*PAGE\s+\d+\s*(?:/\s*\d+)?\s*={2,}",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace only for comparison.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    if not TXT_PATH.exists() or not CSV_PATH.exists():
        print("TXT or CSV file not found.")
        return

    source = TXT_PATH.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    df = pd.read_csv(
        CSV_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    pattern = r"(?im)^\s*Article\s+(\d+)\b"
    matches = list(re.finditer(pattern, source))

    source_articles = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        source_articles[str(int(match.group(1)))] = source[match.start():end]

    csv_articles = {
        row["article_number"].strip(): row["article_text"]
        for _, row in df.iterrows()
    }

    print("=" * 80)
    print("ENGLISH PDPL DIFFERENCE CHECK")
    print("=" * 80)

    for number in sorted(source_articles, key=int):
        if number not in csv_articles:
            print(f"Article {number}: missing from CSV")
            continue

        txt = normalize(source_articles[number])
        csv = normalize(csv_articles[number])

        if txt == csv:
            continue

        print("\n" + "-" * 80)
        print("Article:", number)
        print("TXT length:", len(txt))
        print("CSV length:", len(csv))

        diff = difflib.SequenceMatcher(None, txt, csv)
        changes = 0

        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            if tag == "equal":
                continue

            changes += 1
            print(f"\nDifference type: {tag}")
            print("TXT segment:", repr(txt[max(0, i1-100):min(len(txt), i2+100)]))
            print("CSV segment:", repr(csv[max(0, j1-100):min(len(csv), j2+100)]))

            if changes >= 5:
                print("More differences omitted for this article.")
                break

    print("\nRead-only check completed. No files were modified.")


if __name__ == "__main__":
    main()
