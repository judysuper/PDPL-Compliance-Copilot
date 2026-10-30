from __future__ import annotations
import re


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str, min_chars: int = 25) -> list[str]:
    text = clean_text(text)
    paras = []
    for p in re.split(r"\n\s*\n+", text):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= min_chars:
            paras.append(p)

    # Fallback for pages with no blank lines.
    if len(paras) <= 2:
        alt = []
        for p in re.split(r"(?<=[\.\!\؟:؛])\s+|\n+", text):
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) >= min_chars:
                alt.append(p)
        if len(alt) > len(paras):
            paras = alt
    return paras
