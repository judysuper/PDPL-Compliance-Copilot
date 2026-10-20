
from pathlib import Path
import pandas as pd

expected_files = [
    "pdpl_ar_articles.csv",
    "pdpl_en_articles.csv",
    "implementing_ar_articles.csv",
    "implementing_en_articles.csv",
    "transfer_ar_articles.csv",
    "transfer_en_articles.csv",
]

root = Path(".")
files = {
    p.name: p
    for p in root.rglob("*.csv")
    if p.name in expected_files
}

for filename in expected_files:
    print("\n" + "=" * 50)
    print(filename)

    if filename not in files:
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(files[filename], dtype=str)

    for col in [
        "document_id",
        "language",
        "article_number",
        "article_title",
        "article_text",
        "source_file",
        "source_line_start",
    ]:
        if col not in df.columns:
            print(f"{col}: MISSING COLUMN")
            continue

        values = df[col]
        missing = values.isna() | values.str.strip().eq("")

        print(f"{col}: {missing.sum()} missing / {len(df)}")

    if "article_number" in df.columns:
        numbers = df["article_number"].dropna().str.strip()
        duplicates = numbers[numbers.duplicated(keep=False)].unique()
        print("Duplicate article numbers:", list(duplicates))
