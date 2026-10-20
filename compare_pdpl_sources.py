
from pathlib import Path
import re
import unicodedata
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# 2. Normalize text for comparison only
#    This function does NOT modify source files.
# ============================================================

def normalize_for_comparison(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "").replace("\u200f", "")

    # Remove English page markers anywhere in the text:
    # ===== PAGE 7 / 16 =====
    text = re.sub(
        r"={2,}\s*PAGE\s*\d+\s*(?:/\s*\d+)?\s*={2,}",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove Arabic page markers anywhere in the text:
    # ===== صفحة 3 =====
    text = re.sub(
        r"={2,}\s*صفحة\s*\d+\s*={2,}",
        " ",
        text,
    )

    # Standardize whitespace for comparison
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 3. Split English PDPL TXT into articles
# ============================================================

def extract_english_articles(text):
    text = unicodedata.normalize("NFKC", text)

    # Remove page markers before splitting into articles
    text = re.sub(
        r"={2,}\s*PAGE\s*\d+\s*(?:/\s*\d+)?\s*={2,}",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Match headings such as Article 1, Article 2, ...
    pattern = re.compile(
        r"(?im)^\s*Article\s+(\d+)\b[^\S\r\n]*"
    )

    matches = list(pattern.finditer(text))
    articles = {}

    for index, match in enumerate(matches):
        article_number = match.group(1)

        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        article_text = text[start:end].strip()
        articles[article_number] = article_text

    return articles


# ============================================================
# 4. Compare one TXT source against its CSV
# ============================================================

def compare_english_pdpl(txt_path, csv_path):
    print("\n" + "=" * 80)
    print("ENGLISH PDPL DIFFERENCE CHECK")
    print("=" * 80)

    if not txt_path.exists():
        print(f"ERROR: TXT file not found: {txt_path}")
        return

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # Read TXT without changing the original file
    txt_text = txt_path.read_text(encoding="utf-8-sig")

    # Read CSV without saving or modifying it
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required_columns = {"article_number", "article_text"}

    if not required_columns.issubset(df.columns):
        print("ERROR: CSV is missing required columns.")
        print("Available columns:", list(df.columns))
        return

    txt_articles = extract_english_articles(txt_text)

    csv_articles = {}

    for _, row in df.iterrows():
        number = str(row["article_number"]).strip()
        article_text = str(row["article_text"]).strip()

        if number:
            csv_articles[number] = article_text

    txt_numbers = set(txt_articles.keys())
    csv_numbers = set(csv_articles.keys())

    missing_from_csv = sorted(
        txt_numbers - csv_numbers,
        key=int,
    )

    missing_from_txt = sorted(
        csv_numbers - txt_numbers,
        key=int,
    )

    matched = []
    review = []

    for number in sorted(
        txt_numbers & csv_numbers,
        key=int,
    ):
        txt_normalized = normalize_for_comparison(
            txt_articles[number]
        )
        csv_normalized = normalize_for_comparison(
            csv_articles[number]
        )

        if txt_normalized == csv_normalized:
            matched.append(number)
        else:
            review.append(
                (
                    number,
                    len(txt_normalized),
                    len(csv_normalized),
                    txt_normalized,
                    csv_normalized,
                )
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(f"TXT file: {txt_path.name}")
    print(f"CSV file: {csv_path.name}")
    print(f"TXT articles detected: {len(txt_articles)}")
    print(f"CSV articles detected: {len(csv_articles)}")

    print("\nArticle numbers in TXT but not CSV:")
    print(missing_from_csv)

    print("\nArticle numbers in CSV but not TXT:")
    print(missing_from_txt)

    print(
        f"\nExact normalized matches: "
        f"{len(matched)} / {len(txt_numbers & csv_numbers)}"
    )

    print(f"Articles requiring review: {len(review)}")

    if review:
        print("\n" + "-" * 80)
        print("DETAILS OF REMAINING DIFFERENCES")
        print("-" * 80)

        for number, txt_len, csv_len, txt_text, csv_text in review:
            print(f"\nArticle {number}")
            print(f"TXT normalized length: {txt_len}")
            print(f"CSV normalized length: {csv_len}")

            print("\nTXT preview:")
            print(txt_text[:350])

            print("\nCSV preview:")
            print(csv_text[:350])

            print("\nLength difference:", txt_len - csv_len)

    print("\n" + "=" * 80)

    if (
        len(matched) == 43
        and not missing_from_csv
        and not missing_from_txt
        and not review
    ):
        print("RESULT: PASS - All 43 English PDPL articles match.")
    else:
        print("RESULT: REVIEW - Some differences remain.")

    print("Read-only check completed. No files were modified.")
    print("=" * 80)


# ============================================================
# 5. Run comparison
# ============================================================

if __name__ == "__main__":
    txt_file = RAW_DIR / "pdpl_en.txt"
    csv_file = PROCESSED_DIR / "pdpl_en_articles.csv"

    compare_english_pdpl(txt_file, csv_file)
