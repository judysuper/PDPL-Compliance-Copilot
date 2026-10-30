
import re
from pathlib import Path
import pandas as pd

# ==========================================
# 1. تحديد مسارات المشروع والملفات
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

TXT_PATH = PROJECT_DIR / "data" / "raw" / "implement_ar.txt"
CSV_PATH = PROJECT_DIR / "data" / "processed" / "implementing_ar_articles.csv"

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "implementing_ar_check.txt"

# ==========================================
# 2. تحويل أرقام المواد العربية المكتوبة بالكلمات
#    إلى أرقام من 1 إلى 38
# ==========================================

ARABIC_ARTICLE_NUMBERS = {
    "الأولى": 1,
    "الثانية": 2,
    "الثالثة": 3,
    "الرابعة": 4,
    "الخامسة": 5,
    "السادسة": 6,
    "السابعة": 7,
    "الثامنة": 8,
    "التاسعة": 9,
    "العاشرة": 10,
    "الحادية عشرة": 11,
    "الثانية عشرة": 12,
    "الثالثة عشرة": 13,
    "الرابعة عشرة": 14,
    "الخامسة عشرة": 15,
    "السادسة عشرة": 16,
    "السابعة عشرة": 17,
    "الثامنة عشرة": 18,
    "التاسعة عشرة": 19,
    "العشرون": 20,
    "الحادية والعشرون": 21,
    "الثانية والعشرون": 22,
    "الثالثة والعشرون": 23,
    "الرابعة والعشرون": 24,
    "الخامسة والعشرون": 25,
    "السادسة والعشرون": 26,
    "السابعة والعشرون": 27,
    "الثامنة والعشرون": 28,
    "التاسعة والعشرون": 29,
    "الثلاثون": 30,
    "الحادية والثلاثون": 31,
    "الثانية والثلاثون": 32,
    "الثالثة والثلاثون": 33,
    "الرابعة والثلاثون": 34,
    "الخامسة والثلاثون": 35,
    "السادسة والثلاثون": 36,
    "السابعة والثلاثون": 37,
    "الثامنة والثلاثون": 38,
}

# ==========================================
# 3. التحقق من وجود الملفات
# ==========================================

print("=" * 60)
print("IMPLEMENTING REGULATIONS - ARABIC CHECK")
print("=" * 60)

if not TXT_PATH.exists():
    raise FileNotFoundError(
        f"TXT file not found:\n{TXT_PATH}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV file not found:\n{CSV_PATH}"
    )

# ==========================================
# 4. قراءة الملفات دون تعديلها
# ==========================================

txt_content = TXT_PATH.read_text(encoding="utf-8-sig")

df = pd.read_csv(
    CSV_PATH,
    dtype=str,
    encoding="utf-8-sig"
)

required_columns = {"article_number", "article_text"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required CSV columns: {sorted(missing_columns)}"
    )

# ==========================================
# 5. استخراج أرقام المواد من ملف TXT
# ==========================================

heading_pattern = re.compile(
    r"^\s*المادة\s+(.+?)\s*[:：]\s*(.*?)\s*$"
)

detected = []
unknown_numbers = []

for line in txt_content.splitlines():
    match = heading_pattern.match(line)

    if not match:
        continue

    raw_number = match.group(1).strip()
    normalized_number = raw_number.replace("ـ", "").strip()

    # التعامل مع الأرقام المكتوبة بالأرقام مباشرة
    if normalized_number.isdigit():
        article_number = int(normalized_number)

    # التعامل مع الأرقام المكتوبة بالكلمات
    elif normalized_number in ARABIC_ARTICLE_NUMBERS:
        article_number = ARABIC_ARTICLE_NUMBERS[normalized_number]

    else:
        unknown_numbers.append(raw_number)
        continue

    detected.append(article_number)

# ==========================================
# 6. استخراج أرقام المواد من CSV
# ==========================================

csv_numbers = []

for value in df["article_number"].dropna():
    value = str(value).strip()

    if value.isdigit():
        csv_numbers.append(int(value))

csv_unknown = []

for value in df["article_number"].dropna():
    value = str(value).strip()

    if not value.isdigit():
        csv_unknown.append(value)

# ==========================================
# 7. فحص التكرار والتطابق
# ==========================================

txt_set = set(detected)
csv_set = set(csv_numbers)

txt_duplicates = sorted(
    number for number in txt_set
    if detected.count(number) > 1
)

csv_duplicates = sorted(
    number for number in csv_set
    if csv_numbers.count(number) > 1
)

missing_from_csv = sorted(txt_set - csv_set)
extra_in_csv = sorted(csv_set - txt_set)

expected_numbers = set(range(1, 39))

missing_from_txt_expected = sorted(expected_numbers - txt_set)
missing_from_csv_expected = sorted(expected_numbers - csv_set)

# ==========================================
# 8. تحديد النتيجة النهائية
# ==========================================

passed = all([
    len(detected) == 38,
    len(df) == 38,
    txt_set == expected_numbers,
    csv_set == expected_numbers,
    not txt_duplicates,
    not csv_duplicates,
    not unknown_numbers,
    not csv_unknown,
    not missing_from_csv,
    not extra_in_csv,
])

# ==========================================
# 9. إعداد تقرير الفحص
# ==========================================

report_lines = [
    "IMPLEMENTING REGULATIONS - ARABIC CHECK",
    "=" * 60,
    f"TXT file: {TXT_PATH.name}",
    f"CSV file: {CSV_PATH.name}",
    "",
    f"TXT detected headings: {len(detected)}",
    f"CSV rows: {len(df)}",
    f"Unique article numbers in TXT: {len(txt_set)}",
    f"Unique article numbers in CSV: {len(csv_set)}",
    "",
    f"Numbers in TXT but not CSV: {missing_from_csv}",
    f"Numbers in CSV but not TXT: {extra_in_csv}",
    f"Expected numbers missing from TXT: {missing_from_txt_expected}",
    f"Expected numbers missing from CSV: {missing_from_csv_expected}",
    f"Duplicates in TXT: {txt_duplicates}",
    f"Duplicates in CSV: {csv_duplicates}",
    f"Unknown TXT numbers: {unknown_numbers}",
    f"Unknown CSV numbers: {csv_unknown}",
    "",
    f"Result: {'PASS' if passed else 'REVIEW'}",
    "",
    "Note: This report checks article numbering and structure.",
    "It does not independently verify the legal meaning of the text.",
    "No source files were modified by this script.",
]

report_text = "\n".join(report_lines)

# ==========================================
# 10. عرض النتيجة في Terminal
# ==========================================

print(f"TXT detected headings: {len(detected)}")
print(f"CSV rows: {len(df)}")
print(f"Numbers in TXT but not CSV: {missing_from_csv}")
print(f"Numbers in CSV but not TXT: {extra_in_csv}")
print(f"Duplicates in TXT: {txt_duplicates}")
print(f"Duplicates in CSV: {csv_duplicates}")
print(f"Unknown TXT numbers: {unknown_numbers}")
print(f"Unknown CSV numbers: {csv_unknown}")
print("-" * 60)
print(f"RESULT: {'PASS' if passed else 'REVIEW'}")

# ==========================================
# 11. حفظ التقرير داخل مجلد reports
# ==========================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH.write_text(
    report_text,
    encoding="utf-8"
)

print(f"\nReport saved to:\n{REPORT_PATH}")
print("\nRead-only check completed. No source files were modified.")
