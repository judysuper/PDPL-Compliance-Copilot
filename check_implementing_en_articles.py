
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

txt_path = BASE_DIR / "data" / "raw" / "implementing_en.txt"
csv_path = BASE_DIR / "data" / "processed" / "implementing_en_articles.csv"

# Read source files without modifying them
text = txt_path.read_text(encoding="utf-8-sig")

df = pd.read_csv(
    csv_path,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

print("=" * 75)
print("IMPLEMENTING REGULATIONS - ENGLISH ARTICLE CHECK")
print("=" * 75)

# Detect actual article headings only.
# Requiring ":" prevents matching references such as "Article 15 of the Law".
pattern = re.compile(
    r"(?im)^[ \t]*Article[ \t]+(\d+)[ \t]*:[ \t]*([^\r\n]*)"
)

matches = list(pattern.finditer(text))

print(f"\nTXT detected headings: {len(matches)}")
print(f"CSV rows: {len(df)}")

print("\n--- All detected headings in TXT ---")

txt_numbers = []

for i, match in enumerate(matches, start=1):
    number = match.group(1)
    heading = match.group(0).strip()

    txt_numbers.append(number)

    print(f"{i:02d}. {heading[:180]}")

csv_numbers = df["article_number"].astype(str).str.strip().tolist()

print("\n--- Article numbers in CSV ---")
print(csv_numbers)

print("\n--- Number comparison ---")

txt_set = set(txt_numbers)
csv_set = set(csv_numbers)

print("Numbers detected in TXT but not CSV:")
print(sorted(txt_set - csv_set, key=int))

print("\nNumbers in CSV but not detected in TXT:")
print(sorted(csv_set - txt_set, key=int))

print("\n--- Duplicate numbers detected in TXT ---")
duplicates = sorted(
    {number for number in txt_numbers if txt_numbers.count(number) > 1},
    key=int,
)
print(duplicates)

print("\n--- Duplicate numbers in CSV ---")
csv_duplicates = sorted(
    {number for number in csv_numbers if csv_numbers.count(number) > 1},
    key=int,
)
print(csv_duplicates)

print("\n--- Final result ---")

if (
    len(matches) == 38
    and len(df) == 38
    and txt_set == csv_set
    and not duplicates
    and not csv_duplicates
):
    print("PASS: 38 article headings detected and matched with CSV.")
else:
    print("REVIEW: Differences remain. Check the results above.")

print("\nRead-only check completed. No files were modified.")
