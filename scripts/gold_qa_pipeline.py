"""PDPL Compliance Copilot - Gold Q&A validation and review preparation.
Run from VS Code: python scripts/gold_qa_pipeline.py
Requires: pandas. Never overwrites the gold Q&A dataset or existing human review.
"""
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
import csv
import json
import re
import sys
import pandas as pd

SOURCES = {
    "pdpl_ar": "pdpl_ar_articles.csv",
    "pdpl_en": "pdpl_en_articles.csv",
    "implementing_ar": "implementing_ar_articles.csv",
    "implementing_en": "implementing_en_articles.csv",
    "transfer_ar": "transfer_ar_articles.csv",
    "transfer_en": "transfer_en_articles.csv",
}
REQUIRED_COLUMNS = [
    "question_id", "question", "language", "question_type",
    "reference_answer", "answerable_from_sources", "source_document",
    "article_number", "supporting_text", "expected_behavior",
]


def locate_project():
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd(), here, *here.parents, *Path.cwd().parents]
    for base in candidates:
        if (base / "data" / "processed").is_dir() and (base / "data" / "evaluation").is_dir():
            return base
    raise FileNotFoundError("Project not found. Expected data/processed and data/evaluation.")


def read_csv(path):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    root = locate_project()
    processed = root / "data" / "processed"
    evaluation = root / "data" / "evaluation"
    gold_file = evaluation / "gold_qa_set.csv"
    if not gold_file.exists():
        raise FileNotFoundError(f"Missing {gold_file}")
    qa = read_csv(gold_file)
    missing = [c for c in REQUIRED_COLUMNS if c not in qa.columns]
    if missing:
        raise ValueError(f"Missing Gold Q&A columns: {missing}")

    legal = {}
    source_counts = {}
    for source, filename in SOURCES.items():
        path = processed / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing legal source: {path}")
        df = read_csv(path)
        for col in ["article_number", "article_text"]:
            if col not in df.columns:
                raise ValueError(f"Missing {col} in {filename}")
        legal[source] = df
        source_counts[source] = len(df)

    issues = []
    result_rows = []
    for _, row in qa.iterrows():
        qid = row["question_id"]
        checks = []
        if not re.fullmatch(r"Q\d{3,}", qid):
            checks.append("invalid_question_id")
        if not row["question"].strip():
            checks.append("empty_question")
        if not row["reference_answer"].strip():
            checks.append("empty_reference_answer")
        if row["language"] not in {"ar", "en"}:
            checks.append("invalid_language")
        if row["question_type"] not in {"direct", "scenario", "unanswerable", "analytical", "multi_article"}:
            checks.append("unexpected_question_type")

        answerable = row["answerable_from_sources"].strip().lower()
        if answerable == "true":
            source = row["source_document"].strip()
            number = row["article_number"].strip()
            if source not in legal:
                checks.append("unknown_source")
            else:
                matches = legal[source][legal[source]["article_number"].str.strip() == number]
                if len(matches) != 1:
                    checks.append("missing_or_ambiguous_article")
                elif row["supporting_text"].strip() != matches.iloc[0]["article_text"].strip():
                    checks.append("supporting_text_mismatch")
            if not row["supporting_text"].strip():
                checks.append("missing_supporting_text")
            if row["expected_behavior"].strip() != "answer_with_citation":
                checks.append("incorrect_answer_behavior")
        elif answerable == "false":
            if row["expected_behavior"].strip() != "abstain":
                checks.append("incorrect_abstention_behavior")
            if any(row[k].strip() for k in ("source_document", "article_number", "supporting_text")):
                checks.append("unexpected_unanswerable_citation")
        else:
            checks.append("invalid_answerability")

        for issue in checks:
            issues.append({"question_id": qid, "issue": issue})
        result_rows.append({
            "question_id": qid,
            "automated_check": "PASS" if not checks else "REVIEW",
            "issues": "; ".join(checks),
        })

    if qa["question_id"].duplicated().any():
        for qid in qa.loc[qa["question_id"].duplicated(keep=False), "question_id"].unique():
            issues.append({"question_id": qid, "issue": "duplicate_question_id"})
    if qa["question"].str.strip().duplicated().any():
        issues.append({"question_id": "GENERAL", "issue": "duplicate_question_text"})

    counts = {
        "total": len(qa),
        "arabic": int((qa["language"] == "ar").sum()),
        "english": int((qa["language"] == "en").sum()),
        "scenario": int((qa["question_type"] == "scenario").sum()),
        "unanswerable": int((qa["answerable_from_sources"].str.lower() == "false").sum()),
    }
    minimums = {"total": 60, "arabic": 20, "scenario": 20, "unanswerable": 10}
    for metric, minimum in minimums.items():
        if counts[metric] < minimum:
            issues.append({"question_id": "GENERAL", "issue": f"{metric}_below_{minimum}"})

    # Similarity suggestions are only candidates for human inspection.
    pairs = []
    for i in range(len(qa)):
        q1 = qa.iloc[i]
        for j in range(i + 1, len(qa)):
            q2 = qa.iloc[j]
            if q1["language"] != q2["language"]:
                continue
            normalized1 = re.sub(r"\s+", " ", q1["question"].strip())
            normalized2 = re.sub(r"\s+", " ", q2["question"].strip())
            similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
            if similarity >= 0.74:
                pairs.append({
                    "question_id_1": q1["question_id"],
                    "question_id_2": q2["question_id"],
                    "text_similarity": round(similarity, 3),
                    "human_review_needed": "yes",
                })

    write_csv(pd.DataFrame(result_rows), evaluation / "gold_qa_automated_checks.csv")
    write_csv(pd.DataFrame(issues, columns=["question_id", "issue"]), evaluation / "gold_qa_validation_issues.csv")
    write_csv(pd.DataFrame(pairs, columns=["question_id_1", "question_id_2", "text_similarity", "human_review_needed"]),
              evaluation / "gold_qa_possible_duplicates.csv")

    review_file = evaluation / "gold_qa_review.csv"
    if not review_file.exists():
        review = qa[["question_id", "question", "reference_answer", "source_document", "article_number"]].copy()
        for col in ["legal_accuracy", "citation_support", "question_clarity", "reviewer_id", "review_notes"]:
            review[col] = ""
        write_csv(review, review_file)
        review_msg = "Created blank human review template"
    else:
        review_msg = "Existing human review preserved (not overwritten)"
        reviewed = read_csv(review_file)
        original_ids = set(reviewed["question_id"]) if "question_id" in reviewed else set()
        new_ids = set(qa["question_id"]) - original_ids
        if new_ids:
            issues.append({"question_id": "GENERAL", "issue": f"review_missing_ids:{','.join(sorted(new_ids))}"})

    report = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
        "legal_source_record_counts": source_counts,
        "automated_issue_count": len(issues),
        "possible_similar_question_pairs": len(pairs),
        "review_file_status": review_msg,
        "important_note": "Automated validation does not verify legal correctness, adequacy of cited evidence, source currency, or true unanswerability. Human review is required.",
    }
    (evaluation / "gold_qa_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== GOLD Q&A VALIDATION ===")
    print("Questions:", counts["total"], "| Arabic:", counts["arabic"], "| English:", counts["english"])
    print("Scenarios:", counts["scenario"], "| Unanswerable:", counts["unanswerable"])
    print("Automated issues:", len(issues))
    print("Possible similar pairs:", len(pairs))
    print(review_msg)
    print("Results saved in:", evaluation)
    print("Human legal review is STILL REQUIRED.")
    if issues:
        print("Check gold_qa_validation_issues.csv for details.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
