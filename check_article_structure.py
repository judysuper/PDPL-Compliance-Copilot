
from pathlib import Path
import pandas as pd

root = Path("data/processed")

files = sorted(root.glob("*_articles.csv"))

for path in files:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    print("\n" + "=" * 65)
    print(path.name)

    # فحص تكرار أرقام المواد
    numbers = df["article_number"].str.strip()
    duplicated = df[numbers.duplicated(keep=False)]

    print("Total records:", len(df))
    print("Unique article numbers:", numbers.nunique())
    print("Duplicated article numbers:",
          duplicated["article_number"].tolist())

    # فحص تسلسل الأرقام الرقمي، دون افتراض أن كل وثيقة
    # يجب أن تبدأ من 1 أو أن تكون متصلة بلا استثناء
    numeric = pd.to_numeric(numbers, errors="coerce")
    print("Non-numeric article numbers:",
          numbers[numeric.isna()].tolist())

    # إظهار عناوين النصوص التي لا يوجد لها article_title
    missing_title = df[
        df["article_title"].str.strip().eq("")
    ]

    print("Missing titles:", len(missing_title))

    for _, row in missing_title.head(5).iterrows():
        first_line = (
            row["article_text"].strip().splitlines() or [""]
        )[0]

        print(
            f"Article {row['article_number']} | "
            f"First line: {first_line[:160]}"
        )

# مقارنة أرقام المواد بين اللغتين لكل وثيقة
print("\n" + "=" * 65)
print("AR/EN ARTICLE NUMBER COMPARISON")

for prefix in ["pdpl", "implementing", "transfer"]:
    ar_path = root / f"{prefix}_ar_articles.csv"
    en_path = root / f"{prefix}_en_articles.csv"

    if not ar_path.exists() or not en_path.exists():
        print(prefix, "- missing language pair")
        continue

    ar = pd.read_csv(
        ar_path, dtype=str, keep_default_na=False
    )
    en = pd.read_csv(
        en_path, dtype=str, keep_default_na=False
    )

    ar_nums = set(ar["article_number"].str.strip())
    en_nums = set(en["article_number"].str.strip())

    print(f"\n{prefix}:")
    print("Arabic only:", sorted(ar_nums - en_nums))
    print("English only:", sorted(en_nums - ar_nums))
