
from pathlib import Path
from collections import Counter
import pandas as pd
import re
import unicodedata

# ==========================================
# 1. Paths
# ==========================================

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "transfer_quality_check.txt"

FILES = {
    "Arabic": {
        "txt": RAW / "transfer_ar.txt",
        "csv": PROCESSED / "transfer_ar_articles.csv",
    },
    "English": {
        "txt": RAW / "transfer_en.txt",
        "csv": PROCESSED / "transfer_en_articles.csv",
    },
}

ARABIC_NUMBERS = {
    "الأولى": 1,
    "الثانية": 2,
    "الثالثة": 3,
    "الرابعة": 4,
    "الخامسة": 5,
    "السادسة": 6,
    "السابعة": 7,
    "الثامنة": 8,
    "التاسعة": 9,
}

EXPECTED = set(range(1, 10))
report = []
all_pass = True


def log(message=""):
    print(message)
    report.append(str(message))


def normalize_text(text):
    """Comparison only; never writes back to source files."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page separators in either known format.
    text = re.sub(
        r"(?im)^\s*=+\s*PAGE\s+\d+\s*/\s*\d+\s*=+\s*$",
        " ",
        text,
    )
    text = re.sub(
        r"(?m)^\s*=+\s*صفحة\s+\d+.*?=+\s*$",
        " ",
        text,
    )

    # Ignore layout differences caused by PDF extraction.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_heading_pattern(language):
    if language == "Arabic":
        return re.compile(
            r"(?m)^\s*المادة\s+"
            r"(الأولى|الثانية|الثالثة|الرابعة|الخامسة|"
            r"السادسة|السابعة|الثامنة|التاسعة)"
            r"\s*[:：]\s*"
        )

    return re.compile(
        r"(?im)^\s*Article\s+([0-9]+)\s*[:：]\s*"
    )


def extract_sections(text, language):
    pattern = get_heading_pattern(language)
    matches = list(pattern.finditer(text))
    sections = {}

    for index, match in enumerate(matches):
        if language == "Arabic":
            number = ARABIC_NUMBERS[match.group(1)]
        else:
            number = int(match.group(1))

        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        if number in sections:
            sections[number]["duplicate"] = True
        else:
            sections[number] = {
                "text": text[start:end].strip(),
                "duplicate": False,
            }

    return sections


def compare_text(source_text, csv_text):
    source_normalized = normalize_text(source_text)
    csv_normalized = normalize_text(csv_text)

    if source_normalized == csv_normalized:
        return "EXACT_MATCH"

    if csv_normalized and csv_normalized in source_normalized:
        return "CSV_TEXT_IS_SUBSTRING_REVIEW"

    if source_normalized and source_normalized in csv_normalized:
        return "SOURCE_TEXT_IS_SUBSTRING_REVIEW"

    return "TEXT_DIFFERENCE_REVIEW"


# ==========================================
# 2. Run checks
# ==========================================

log("=" * 70)
log("TRANSFER REGULATION - QUALITY CHECK")
log("=" * 70)
log("Read-only: original TXT and CSV files will not be modified.")
log("")

for language, paths in FILES.items():
    log("-" * 70)
    log(f"LANGUAGE: {language}")
    log("-" * 70)

    txt_path = paths["txt"]
    csv_path = paths["csv"]

    if not txt_path.exists() or not csv_path.exists():
        log(f"ERROR: Missing file: {txt_path if not txt_path.exists() else csv_path}")
        all_pass = False
        continue

    source = txt_path.read_text(encoding="utf-8-sig")
    df = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required = {"article_number", "article_text"}
    missing_columns = required - set(df.columns)

    if missing_columns:
        log(f"ERROR: Missing CSV columns: {sorted(missing_columns)}")
        all_pass = False
        continue

    # Extract numbered article sections from the source TXT.
    source_sections = extract_sections(source, language)

    # Validate CSV article numbers.
    csv_numbers = []
    invalid_numbers = []

    for value in df["article_number"]:
        normalized = unicodedata.normalize("NFKC", str(value)).strip()

        if re.fullmatch(r"\d+", normalized):
            csv_numbers.append(int(normalized))
        else:
            invalid_numbers.append(str(value))

    duplicates_csv = sorted(
        number
        for number, count in Counter(csv_numbers).items()
        if count > 1
    )

    duplicates_txt = sorted(
        number
        for number, section in source_sections.items()
        if section["duplicate"]
    )

    source_numbers = set(source_sections)
    csv_number_set = set(csv_numbers)

    log(f"TXT detected headings: {len(source_sections)}")
    log(f"CSV rows: {len(df)}")
    log(f"TXT article numbers: {sorted(source_numbers)}")
    log(f"CSV article numbers: {sorted(csv_number_set)}")
    log(f"Numbers missing from CSV: {sorted(source_numbers - csv_number_set)}")
    log(f"Numbers missing from TXT: {sorted(csv_number_set - source_numbers)}")
    log(f"Duplicate TXT headings: {duplicates_txt}")
    log(f"Duplicate CSV numbers: {duplicates_csv}")
    log(f"Invalid CSV numbers: {invalid_numbers}")

    empty_text = [
        int(index) + 2
        for index, value in enumerate(df["article_text"])
        if not str(value).strip()
    ]
    log(f"CSV rows with empty article_text: {empty_text}")

    # Compare every CSV article against its source section.
    log("")
    log("ARTICLE TEXT COMPARISON")

    comparisons = {}

    for _, row in df.iterrows():
        normalized_number = unicodedata.normalize(
            "NFKC", str(row["article_number"])
        ).strip()

        if not normalized_number.isdigit():
            continue

        number = int(normalized_number)

        if number not in source_sections:
            comparisons[number] = "SOURCE_HEADING_NOT_FOUND"
            continue

        status = compare_text(
            source_sections[number]["text"],
            row["article_text"],
        )
        comparisons[number] = status

        log(f"Article {number}: {status}")

        if status != "EXACT_MATCH":
            source_preview = normalize_text(
                source_sections[number]["text"]
            )[:180]
            csv_preview = normalize_text(row["article_text"])[:180]

            log(f"  TXT preview: {source_preview}")
            log(f"  CSV preview: {csv_preview}")

    exact_matches = sum(
        status == "EXACT_MATCH"
        for status in comparisons.values()
    )

    review_count = sum(
        status != "EXACT_MATCH"
        for status in comparisons.values()
    )

    log("")
    log(f"Exact text matches: {exact_matches}")
    log(f"Articles requiring review: {review_count}")

    passed = all([
        len(df) == 9,
        source_numbers == EXPECTED,
        csv_number_set == EXPECTED,
        len(csv_numbers) == 9,
        not duplicates_txt,
        not duplicates_csv,
        not invalid_numbers,
        not empty_text,
        len(comparisons) == 9,
        exact_matches == 9,
    ])

    log(f"LANGUAGE RESULT: {'PASS' if passed else 'REVIEW'}")

    if not passed:
        all_pass = False

    log("")

log("=" * 70)
log("FINAL RESULT")
log("=" * 70)
log(f"Overall: {'PASS' if all_pass else 'REVIEW'}")
log("Comparison normalizes formatting only; it does not edit legal text.")
log("A text difference must be reviewed before any correction.")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print(f"\nReport saved to: {REPORT_PATH}")
print("No source files were modified.")
