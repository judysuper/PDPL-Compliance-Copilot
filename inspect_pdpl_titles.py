
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

files = {
    "English": RAW_DIR / "pdpl_en.txt",
    "Arabic": RAW_DIR / "pdpl_ar.txt",
}

for language, path in files.items():
    print("\n" + "=" * 70)
    print(f"{language} PDPL - Inspecting article headings")
    print("=" * 70)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    text = path.read_text(encoding="utf-8-sig")

    if language == "English":
        pattern = re.compile(
            r"(?im)^\s*Article\s+(\d+)\b[^\S\r\n]*(.*)$"
        )
    else:
        pattern = re.compile(
            r"(?m)^\s*المادة\s+([^\s:：]+)\s*[:：]?\s*(.*)$"
        )

    matches = list(pattern.finditer(text))

    for match in matches[:10]:
        number = match.group(1)
        heading_remainder = match.group(2).strip()

        start = match.end()
        next_start = (
            matches[matches.index(match) + 1].start()
            if matches.index(match) + 1 < len(matches)
            else len(text)
        )

        following_text = text[start:next_start].strip()
        following_text = re.sub(
            r"(?m)^\s*=+\s*(?:PAGE\s*)?\d+.*?=+\s*$",
            "",
            following_text,
            flags=re.IGNORECASE,
        ).strip()

        print(f"\nArticle {number}")
        print(f"Text on heading line: {heading_remainder[:200]}")
        print(f"Following text: {following_text[:250]}")

    print(f"\nTotal headings detected: {len(matches)}")
