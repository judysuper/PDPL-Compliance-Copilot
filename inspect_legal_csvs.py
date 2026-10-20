
from pathlib import Path
import csv
import json

# ابحث عن الملفات القانونية داخل مجلد المشروع
ROOT = Path(".")
files = sorted(ROOT.rglob("*_articles.csv"))

if not files:
    print("ERROR: لم يتم العثور على ملفات *_articles.csv")
    raise SystemExit(1)

print(f"Found {len(files)} CSV files")

for path in files:
    print("\n" + "=" * 70)
    print(f"FILE: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)

            # محاولة اكتشاف الفاصل بين الأعمدة
            try:
                dialect = csv.Sniffer().sniff(
                    sample, delimiters=",;\t|"
                )
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)
            headers = reader.fieldnames or []
            rows = list(reader)

        print("Columns:", json.dumps(headers, ensure_ascii=False))
        print("Data rows:", len(rows))

        # فحص الصفوف الفارغة والتكرارات
        empty_rows = 0
        duplicate_rows = 0
        seen = set()

        for row in rows:
            values = tuple(
                (value or "").strip()
                for value in row.values()
            )

            if not any(values):
                empty_rows += 1

            if values in seen:
                duplicate_rows += 1
            seen.add(values)

        print("Completely empty rows:", empty_rows)
        print("Duplicate full rows:", duplicate_rows)

        # عرض ملخص الأعمدة والنصوص الفارغة
        for col in headers:
            non_empty = sum(
                1 for row in rows
                if (row.get(col) or "").strip()
            )
            print(
                f"  {col}: {non_empty}/{len(rows)} non-empty"
            )

        print("\nFirst 2 records:")
        for row in rows[:2]:
            print(json.dumps(row, ensure_ascii=False))

    except Exception as e:
        print(f"ERROR reading file: {e}")
