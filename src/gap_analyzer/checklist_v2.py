from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    requirement: str
    requirement_type: str
    source_document: str
    article_number: str
    legal_reference: str
    verification_question: str
    evidence_needed: str
    notes: str


def load_requirements(path: str | Path) -> list[Requirement]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checklist not found: {path}")

    out: list[Requirement] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "requirement_id", "requirement", "requirement_type",
            "source_document", "article_number", "legal_reference",
            "verification_question", "evidence_needed", "notes"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Checklist missing columns: {sorted(missing)}")

        for row in reader:
            rid = (row.get("requirement_id") or "").strip()
            if not rid:
                continue
            out.append(Requirement(
                requirement_id=rid,
                requirement=(row.get("requirement") or "").strip(),
                requirement_type=(row.get("requirement_type") or "").strip(),
                source_document=(row.get("source_document") or "").strip(),
                article_number=(row.get("article_number") or "").strip(),
                legal_reference=(row.get("legal_reference") or "").strip(),
                verification_question=(row.get("verification_question") or "").strip(),
                evidence_needed=(row.get("evidence_needed") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            ))

    if len(out) != 15:
        raise ValueError(f"Expected 15 PDPL requirements, found {len(out)}")
    return out
