
from pathlib import Path
import pandas as pd

# مجلد ملفات CSV
DATA_DIR = Path("data/processed")

FILES = [
    DATA_DIR / "pdpl_ar_articles.csv",
    DATA_DIR / "pdpl_en_articles.csv",
]

REQUIRED_COLUMNS = [
    "article_number",
    "article_title",
    "article_text",
]


def inspect_file(file_path):
    print("\n" + "=" * 90)
    print(f"FILE: {file_path.name}")
    print("=" * 90)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    # قراءة فقط: لا يتم تعديل الملف
    df = pd.read_csv(
        file_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    # التحقق من وجود الأعمدة المطلوبة
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing columns:", missing_columns)
        print("Available columns:", df.columns.tolist())
        return

    print("Total records:", len(df))
    print("Columns:", df.columns.tolist())

    # فحص العناوين الفارغة
    empty_titles = df["article_title"].str.strip().eq("").sum()
    print("Missing titles:", empty_titles)

    # عرض جميع المواد
    for index, row in df.iterrows():
        article_number = row["article_number"].strip()
        article_title = row["article_title"].strip()
        article_text = row["article_text"]

        # إظهار الأسطر بطريقة واضحة
        preview = article_text[:300]

        print("\n" + "-" * 90)
        print(f"Record: {index + 1}")
        print(f"Article number: {article_number}")
        print(f"Article title: {article_title or '[EMPTY]'}")
        print(f"Article text (first 300 characters):\n{preview}")
        print(f"Article text length: {len(article_text)} characters")

    print("\n" + "=" * 90)
    print(f"Finished inspecting: {file_path.name}")
    print("No files were modified.")


def main():
    for file_path in FILES:
        inspect_file(file_path)


if __name__ == "__main__":
    main()
