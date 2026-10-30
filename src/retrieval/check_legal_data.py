
from pathlib import Path
import pandas as pd

data_dir = Path("../../data/processed")

for file in sorted(data_dir.glob("*.csv")):
    df = pd.read_csv(file).fillna("")
    text = df["article_text"].astype(str).str.strip()

    empty = (text == "").sum()
    duplicates = text.duplicated().sum()
    missing_numbers = (
        df["article_number"].astype(str).str.strip() == ""
    ).sum()

    print(f"\n{'=' * 45}")
    print(f"File: {file.name}")
    print(f"Rows: {len(df)}")
    print(f"Empty texts: {empty}")
    print(f"Duplicate texts: {duplicates}")
    print(f"Missing article numbers: {missing_numbers}")

    if len(text) > 0:
        print(f"Sample text: {text.iloc[0][:250]}")
