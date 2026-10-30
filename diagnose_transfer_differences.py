
from pathlib import Path
import pandas as pd
import re
import unicodedata
import difflib

# ==========================================
# 1. Project paths
# ==========================================

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "transfer_differences_detail.txt"

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

EXPECTED_ARTICLES = set(range(1, 10))
report = []


# ==========================================
# 2. Reporting
# ==========================================

def log(message=""):
    print(message)
    report.append(str(message))


# ==========================================
# 3. Normalize formatting for comparison only
# ==========================================

def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove standalone page markers.
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

    # Remove Arabic document classification footer.
    text = re.sub(
        r"تصنيف\s+الوثيقة\s*:\s*عام\s*\d*\s*"
        r"(?:\[شعار:.*?\])?",
        " ",
        text,
    )

    # Remove English document classification footer.
    text = re.sub(
        r"\b\d+\s+Public\s*:\s*Classification\s*Document\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove known back-cover descriptions.
    text = re.sub(
        r"\[صفحة الغلاف الخلفي:.*?\]\s*(?:\[شعار:.*?\])?",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================
# 4. Article heading detection
# ==========================================

def heading_pattern(language):
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
    pattern = heading_pattern(language)
    matches = list(pattern.finditer(text))
    sections = {}
    duplicates = []

    for index, match in enumerate(matches):
        if language == "Arabic":
            number = ARABIC_NUMBERS[match.group(1)]
        else:
            number = int(match.group(1))

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        if number in sections:
            duplicates.append(number)
        else:
            sections[number] = text[match.start():end].strip()

    return sections, sorted(set(duplicates))


# ==========================================
# 5. Detailed text comparison
# ==========================================

def first_difference(a, b):
    for index, (char_a, char_b) in enumerate(zip(a, b)):
        if char_a != char_b:
            return index

    if len(a) != len(b):
        return min(len(a), len(b))

    return None


def excerpt(text, position, radius=120):
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return text[start:end]


def compare_text(source_text, csv_text):
    source_normalized = normalize(source_text)
    csv_normalized = normalize(csv_text)

    if source_normalized == csv_normalized:
        return "EXACT_MATCH", source_normalized, csv_normalized

    if csv_normalized and csv_normalized in source_normalized:
        return (
            "CSV_TEXT_IS_SUBSTRING_REVIEW",
            source_normalized,
            csv_normalized,
        )

    if source_normalized and source_normalized in csv_normalized:
        return (
            "SOURCE_TEXT_IS_SUBSTRING_REVIEW",
            source_normalized,
            csv_normalized,
        )

    return "TEXT_DIFFERENCE_REVIEW", source_normalized, csv_normalized


# ==========================================
# 6. Inspect each language
# ==========================================

overall_pass = True

for language, paths in FILES.items():
    log("=" * 75)
    log(f"{language.upper()} - DETAILED TEXT DIFFERENCES")
    log("=" * 75)

    txt_path = paths["txt"]
    csv_path = paths["csv"]

    if not txt_path.exists():
        log(f"ERROR: TXT file not found: {txt_path}")
        overall_pass = False
        continue

    if not csv_path.exists():
        log(f"ERROR: CSV file not found: {csv_path}")
        overall_pass = False
        continue

    source = txt_path.read_text(encoding="utf-8-sig")

    df = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required_columns = {"article_number", "article_text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        log(f"ERROR: Missing CSV columns: {sorted(missing_columns)}")
        overall_pass = False
        continue

    source_sections, duplicate_source_numbers = extract_sections(
        source, language
    )

    csv_numbers = []
    invalid_numbers = []

    for value in df["article_number"]:
        normalized_number = unicodedata.normalize(
            "NFKC", str(value)
        ).strip()

        if re.fullmatch(r"\d+", normalized_number):
            csv_numbers.append(int(normalized_number))
        else:
            invalid_numbers.append(str(value))

    duplicate_csv_numbers = sorted(
        number
        for number, count in __import__("collections").Counter(
            csv_numbers
        ).items()
        if count > 1
    )

    source_number_set = set(source_sections)
    csv_number_set = set(csv_numbers)

    empty_text_rows = [
        index + 2
        for index, value in enumerate(df["article_text"])
        if not str(value).strip()
    ]

    log(f"TXT headings detected: {len(source_sections)}")
    log(f"CSV rows: {len(df)}")
    log(f"TXT article numbers: {sorted(source_number_set)}")
    log(f"CSV article numbers: {sorted(csv_number_set)}")
    log(f"Numbers in TXT but not CSV: {sorted(source_number_set - csv_number_set)}")
    log(f"Numbers in CSV but not TXT: {sorted(csv_number_set - source_number_set)}")
    log(f"Duplicate TXT headings: {duplicate_source_numbers}")
    log(f"Duplicate CSV article numbers: {duplicate_csv_numbers}")
    log(f"Invalid CSV article numbers: {invalid_numbers}")
    log(f"Empty article_text rows: {empty_text_rows}")
    log("")

    exact_matches = 0
    review_count = 0

    for _, row in df.iterrows():
        number_text = unicodedata.normalize(
            "NFKC", str(row["article_number"])
        ).strip()

        if not number_text.isdigit():
            continue

        number = int(number_text)

        if number not in source_sections:
            log(f"Article {number}: SOURCE_HEADING_NOT_FOUND")
            review_count += 1
            continue

        status, source_normalized, csv_normalized = compare_text(
            source_sections[number],
            row["article_text"],
        )

        log(f"Article {number}: {status}")

        if status == "EXACT_MATCH":
            exact_matches += 1
            continue

        review_count += 1
        position = first_difference(source_normalized, csv_normalized)

        log(f"  TXT normalized length: {len(source_normalized)}")
        log(f"  CSV normalized length: {len(csv_normalized)}")
        log(
            "  Length difference (TXT - CSV): "
            f"{len(source_normalized) - len(csv_normalized)}"
        )

        if position is not None:
            log(f"  First differing position: {position}")
            log("  TXT around difference:")
            log(f"    {excerpt(source_normalized, position)}")
            log("  CSV around difference:")
            log(f"    {excerpt(csv_normalized, position)}")

        if (
            source_normalized not in csv_normalized
            and csv_normalized not in source_normalized
        ):
            matcher = difflib.SequenceMatcher(
                None,
                source_normalized,
                csv_normalized,
                autojunk=False,
            )

            changes = [
                item
                for item in matcher.get_opcodes()
                if item[0] != "equal"
            ]

            log("  First 3 differing regions:")

            for tag, i1, i2, j1, j2 in changes[:3]:
                log(f"    Change type: {tag}")
                log(
                    "    TXT: "
                    + source_normalized[
                        max(0, i1 - 50):min(len(source_normalized), i2 + 50)
                    ]
                )
                log(
                    "    CSV: "
                    + csv_normalized[
                        max(0, j1 - 50):min(len(csv_normalized), j2 + 50)
                    ]
                )

    log("")
    log(f"Exact matches: {exact_matches}")
    log(f"Articles needing review: {review_count}")

    language_pass = all([
        len(df) == 9,
        source_number_set == EXPECTED_ARTICLES,
        csv_number_set == EXPECTED_ARTICLES,
        not duplicate_source_numbers,
        not duplicate_csv_numbers,
        not invalid_numbers,
        not empty_text_rows,
        exact_matches == 9,
        review_count == 0,
    ])

    log(f"LANGUAGE RESULT: {'PASS' if language_pass else 'REVIEW'}")

    if not language_pass:
        overall_pass = False

    log("")


# ==========================================
# 7. Final result and save report
# ==========================================

log("=" * 75)
log("FINAL RESULT")
log("=" * 75)
log(f"Overall: {'PASS' if overall_pass else 'REVIEW'}")
log("Comparison uses normalized text for inspection only.")
log("A REVIEW result does not automatically mean legal text is missing.")
log("No original TXT or CSV files were modified.")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print(f"\nDetailed report saved to:\n{REPORT_PATH}")
