from __future__ import annotations
from pathlib import Path
import csv
import json
import sys

from checklist_v2 import load_requirements
from analyzer_v2 import GapAnalyzerV2

try:
    from classifier_v2 import XLMRClassifierAdapter
except Exception:
    XLMRClassifierAdapter = None


ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = ROOT / "data" / "pdpl_checklist" / "member1_pdpl_compliance_checklist.csv"
RAW_DIR = ROOT / "data" / "local_policies" / "raw"
OUT_ROOT = ROOT / "reports" / "gap_analyzer_v2" / "evaluation"
MODEL_DIR = ROOT / "models" / "xlm_roberta_best"


def find_policy_file(name: str) -> Path:
    candidates = [
        RAW_DIR / f"{name}_ar.txt",
        RAW_DIR / f"{name}_en.txt",
        RAW_DIR / f"{name}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No raw policy text for {name}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/gap_analyzer/run_policy_analysis_v2.py <policy_name>")
        raise SystemExit(2)

    name = sys.argv[1].strip().lower()
    policy_path = find_policy_file(name)
    text = policy_path.read_text(encoding="utf-8", errors="replace")

    requirements = load_requirements(CHECKLIST)

    classifier = None
    if XLMRClassifierAdapter is not None and MODEL_DIR.exists():
        try:
            classifier = XLMRClassifierAdapter(MODEL_DIR)
            print(f"Loaded XLM-R classifier from: {MODEL_DIR}")
        except Exception as e:
            print(f"WARNING: classifier could not be loaded; rule layer only. Reason: {e}")

    analyzer = GapAnalyzerV2(requirements, classifier=classifier)
    results = analyzer.analyze(text)

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{name}_gap_analysis_v2.json"
    csv_path = out_dir / f"{name}_gap_analysis_v2.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    fields = list(results[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    counts = {"Covered": 0, "Partial": 0, "Missing": 0}
    for r in results:
        counts[r["status"]] += 1

    print("=" * 70)
    print(f"SUMMARY - {name.upper()} - GAP ANALYZER V2")
    print("=" * 70)
    print("Total requirements:", len(results))
    print("Covered:", counts["Covered"])
    print("Partial:", counts["Partial"])
    print("Missing:", counts["Missing"])
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
