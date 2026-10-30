
from pathlib import Path
import pandas as pd

# عدّلي المسار حسب مكان ملف القوانين عندكم
DATA_DIR = Path("../../data/processed")

for file_path in DATA_DIR.iterdir():
    if file_path.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
        continue

    print("\n" + "=" * 60)
    print("File:", file_path.name)

    try:
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

        print("\nMissing values:")
        print(df.isna().sum().to_dict())

    except Exception as error:
        print("Could not read file:", error)
