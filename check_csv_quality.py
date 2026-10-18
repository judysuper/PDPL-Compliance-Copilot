
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
import unicodedata
import pandas as pd

# ==================================================
# 1. Project paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "csv_quality_report.txt"

FILES = {
    "implementing_ar": "implementing_ar_articles.csv",
    "implementing_en": "implementing_en_articles.csv",
    "pdpl_ar": "pdpl_ar_articles.csv",
    "pdpl_en": "pdpl_en_articles.csv",
    "transfer_ar": "transfer_ar_articles.csv",
    "transfer_en": "transfer_en_articles.csv",
}

EXPECTED_COUNTS = {
    "implementing_ar": 38,
    "implementing_en": 38,
    "pdpl_ar": 43,
    "pdpl_en": 43,
    "transfer_ar": 9,
    "transfer_en": 9,
}

REQUIRED_COLUMNS = [
    "document_id",
    "language",
    "article_number",
    "article_title",
    "article_text",
    "source_file",
    "source_line_start",
]

PAIRS = [
    ("implementing_ar", "implementing_en"),
    ("pdpl_ar", "pdpl_en"),
    ("transfer_ar", "transfer_en"),
]

# ==================================================
# 2. Helper functions
# ==================================================

def is_blank(value):
    return str(value).strip() == ""


def normalize_number(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()

    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]

    if value.isdigit():
        return int(value)

    return None


def log(message=""):
    print(message)
    report.append(str(message))


# ==================================================
# 3. Initialize report
# ==================================================

report = []
dataframes = {}
file_results = {}

log("=" * 72)
log("PDPL COMPLIANCE COPILOT")
log("CSV QUALITY CHECK REPORT")
log("=" * 72)
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("Mode: READ-ONLY")
log("Original CSV and TXT files will not be modified.")

# ==================================================
# 4. Check all six CSV files
# ==================================================

for key, filename in FILES.items():

    log("\n" + "=" * 72)
    log(f"FILE: {filename}")
    log("=" * 72)

    path = DATA_DIR / filename
    issues = []
    warnings = []

    if not path.exists():
        log(f"ERROR: File not found: {path}")
        file_results[key] = {
            "status": "ERROR",
            "issues": ["File not found"],
            "warnings": [],
        }
        continue

    try:
        df = pd.read_csv(
            path,
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    except Exception as exc:
        log(f"ERROR: Could not read file: {exc}")
        file_results[key] = {
            "status": "ERROR",
            "issues": [str(exc)],
            "warnings": [],
        }
        continue

    dataframes[key] = df

    # ----------------------------------------------
    # A. File structure
    # ----------------------------------------------

    log(f"Rows: {len(df)}")
    log(f"Columns: {len(df.columns)}")
    log(f"Column names: {list(df.columns)}")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in REQUIRED_COLUMNS
    ]

    log(f"Missing required columns: {missing_columns}")
    log(f"Additional columns: {extra_columns}")

    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")

    expected_rows = EXPECTED_COUNTS[key]

    if len(df) != expected_rows:
        issues.append(
            f"Expected {expected_rows} rows, found {len(df)}"
        )

    # ----------------------------------------------
    # B. Empty values
    # ----------------------------------------------

    empty_counts = {
        col: sum(is_blank(value) for value in df[col])
        for col in df.columns
    }

    empty_counts = {
        col: count
        for col, count in empty_counts.items()
        if count > 0
    }

    log(f"Columns with empty values: {empty_counts}")

    for col in ("article_number", "article_text"):
        if col in df.columns:
            empty_rows = [
                index + 2
                for index, value in enumerate(df[col])
                if is_blank(value)
            ]

            if empty_rows:
                issues.append(
                    f"Empty {col} at CSV lines: {empty_rows[:20]}"
                )

    if "article_title" in df.columns:
        empty_titles = sum(
            is_blank(value) for value in df["article_title"]
        )

        if empty_titles:
            if key in ("pdpl_ar", "pdpl_en"):
                warnings.append(
                    f"{empty_titles} empty article titles; "
                    "review whether titles exist in the source."
                )
            else:
                warnings.append(
                    f"{empty_titles} empty article titles."
                )

    # ----------------------------------------------
    # C. Duplicate rows
    # ----------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    log(f"Duplicate full rows: {duplicate_rows}")

    if duplicate_rows:
        issues.append(
            f"Duplicate full rows: {duplicate_rows}"
        )

    # ----------------------------------------------
    # D. Article number validation
    # ----------------------------------------------

    if "article_number" in df.columns:
        normalized_numbers = [
            normalize_number(value)
            for value in df["article_number"]
        ]

        invalid_numbers = [
            str(value)
            for value, number in zip(
                df["article_number"], normalized_numbers
            )
            if not is_blank(value) and number is None
        ]

        valid_numbers = [
            number
            for number in normalized_numbers
            if number is not None
        ]

        counts = Counter(valid_numbers)

        duplicate_numbers = sorted(
            number
            for number, count in counts.items()
            if count > 1
        )

        log(f"Unique article numbers: {len(set(valid_numbers))}")
        log(f"Duplicate article numbers: {duplicate_numbers}")
        log(f"Invalid article number values: {invalid_numbers}")

        if duplicate_numbers:
            issues.append(
                f"Duplicate article numbers: {duplicate_numbers}"
            )

        if invalid_numbers:
            issues.append(
                f"Invalid article number values: {invalid_numbers}"
            )

        expected_numbers = set(range(1, expected_rows + 1))
        actual_numbers = set(valid_numbers)

        missing_numbers = sorted(expected_numbers - actual_numbers)
        unexpected_numbers = sorted(actual_numbers - expected_numbers)

        log(f"Missing expected article numbers: {missing_numbers}")
        log(f"Unexpected article numbers: {unexpected_numbers}")

        if missing_numbers:
            issues.append(
                f"Missing article numbers: {missing_numbers}"
            )

        if unexpected_numbers:
            issues.append(
                f"Unexpected article numbers: {unexpected_numbers}"
            )

    # ----------------------------------------------
    # E. Text length checks
    # ----------------------------------------------

    if "article_text" in df.columns:
        lengths = df["article_text"].astype(str).str.len()

        if len(lengths):
            log(f"Shortest article text: {int(lengths.min())} characters")
            log(f"Longest article text: {int(lengths.max())} characters")
            log(f"Average article text: {lengths.mean():.1f} characters")

        short_articles = []

        for index, row in df.iterrows():
            text = str(row["article_text"])

            if text.strip() and len(text.strip()) < 50:
                short_articles.append({
                    "CSV line": index + 2,
                    "article_number": row.get("article_number", ""),
                    "characters": len(text.strip()),
                })

        log(f"Very short article candidates (<50 chars): {short_articles}")

        if short_articles:
            warnings.append(
                f"{len(short_articles)} very short article candidates "
                "require manual review."
            )

    # ----------------------------------------------
    # F. Suspicious text markers
    # ----------------------------------------------

    suspicious = {
        "NUL characters": [],
        "Replacement characters": [],
        "HTML entities": [],
    }

    text_columns = [
        col for col in ("article_title", "article_text")
        if col in df.columns
    ]

    for index, row in df.iterrows():
        combined_text = " ".join(
            str(row[col]) for col in text_columns
        )

        csv_line = index + 2

        if "\x00" in combined_text:
            suspicious["NUL characters"].append(csv_line)

        if "\ufffd" in combined_text:
            suspicious["Replacement characters"].append(csv_line)

        if re.search(
            r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);",
            combined_text,
        ):
            suspicious["HTML entities"].append(csv_line)

    log(f"Suspicious text markers: {suspicious}")

    for marker, lines in suspicious.items():
        if lines:
            issues.append(
                f"{marker} at CSV lines: {lines[:20]}"
            )

    # ----------------------------------------------
    # G. Metadata checks
    # ----------------------------------------------

    for col in ("document_id", "language", "source_file"):
        if col in df.columns:
            blank_count = sum(
                is_blank(value) for value in df[col]
            )

            if blank_count:
                issues.append(
                    f"Missing {col} values: {blank_count}"
                )

    # ----------------------------------------------
    # H. Per-file result
    # ----------------------------------------------

    status = "PASS" if not issues else "REVIEW"

    log(f"Issues: {issues}")
    log(f"Warnings: {warnings}")
    log(f"FILE RESULT: {status}")

    file_results[key] = {
        "status": status,
        "issues": issues,
        "warnings": warnings,
    }


# ==================================================
# 5. Compare Arabic and English article numbers
# ==================================================

log("\n" + "=" * 72)
log("ARABIC / ENGLISH ARTICLE NUMBER COMPARISON")
log("=" * 72)

for ar_key, en_key in PAIRS:

    log(f"\nPair: {ar_key} / {en_key}")

    if ar_key not in dataframes or en_key not in dataframes:
        log("PAIR RESULT: REVIEW - one or both files unavailable")
        continue

    ar_df = dataframes[ar_key]
    en_df = dataframes[en_key]

    ar_numbers = {
        normalize_number(value)
        for value in ar_df["article_number"]
    }

    en_numbers = {
        normalize_number(value)
        for value in en_df["article_number"]
    }

    ar_numbers.discard(None)
    en_numbers.discard(None)

    ar_only = sorted(ar_numbers - en_numbers)
    en_only = sorted(en_numbers - ar_numbers)

    pair_pass = (
        not ar_only
        and not en_only
        and len(ar_df) == len(en_df)
    )

    log(f"Arabic rows: {len(ar_df)}")
    log(f"English rows: {len(en_df)}")
    log(f"Numbers only in Arabic: {ar_only}")
    log(f"Numbers only in English: {en_only}")
    log(f"PAIR RESULT: {'PASS' if pair_pass else 'REVIEW'}")


# ==================================================
# 6. Final summary
# ==================================================

log("\n" + "=" * 72)
log("FINAL QUALITY SUMMARY")
log("=" * 72)

for key, result in file_results.items():
    log(
        f"{key}: {result['status']} | "
        f"Issues: {len(result['issues'])} | "
        f"Warnings: {len(result['warnings'])}"
    )

error_count = sum(
    len(result["issues"])
    for result in file_results.values()
)

warning_count = sum(
    len(result["warnings"])
    for result in file_results.values()
)

log(f"Total file-level issues: {error_count}")
log(f"Total file-level warnings: {warning_count}")
log("")
log("IMPORTANT:")
log("- This is a structural and text-quality screening.")
log("- It does not independently verify legal meaning.")
log("- Warnings require review and do not automatically mean errors.")
log("- No source files were modified.")

# ==================================================
# 7. Save report
# ==================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 72)
print("QUALITY CHECK COMPLETED")
print("=" * 72)
print(f"Report saved to:\n{REPORT_PATH}")
print("Original CSV and TXT files were not modified.")
