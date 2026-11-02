"""Compare the team's GapAnalyzerV2 on the 10 Gold policies: rules only vs rules + XLM-R (several score cut-offs).

Run from the project root:
    python scripts/compare_rules_vs_classifier.py
    python scripts/compare_rules_vs_classifier.py --min-scores 0.3 0.5 0.7
Prints accuracy, macro-F1, over-coverage (predicted Covered while Gold is not) and per-requirement accuracy.
Saves per-decision predictions to reports/compare_rules_vs_classifier.csv.

NOTE: these 10 policies are the development set (the rules were written while looking at them).
Do not report these numbers as final test performance.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src" / "gap_analyzer", ROOT / "app"):
    sys.path.insert(0, str(p))

import gap_service as gs                               # noqa: E402
from analyzer_v2 import GapAnalyzerV2                  # noqa: E402
from checklist_v2 import load_requirements             # noqa: E402
from rules_v2 import OPP_CATEGORY_MAP                  # noqa: E402

CHECKLIST = ROOT / "data" / "pdpl_checklist" / "member1_pdpl_compliance_checklist.csv"
RAW = ROOT / "data" / "local_policies" / "raw"
GOLD = ROOT / "data" / "local_policies" / "gold" / "pdpl_local_policies_gold.csv"
MODEL_DIR = ROOT / "models" / "xlm_roberta_best"
LABELS = ["Covered", "Partial", "Missing"]


class ThresholdAnalyzer(GapAnalyzerV2):
    """Same decisions as GapAnalyzerV2._status, with the classifier cut-off (0.50 in the original) as a parameter."""

    def __init__(self, requirements, classifier=None, min_score=0.50):
        super().__init__(requirements, classifier)
        self.min_score = min_score

    def _status(self, requirement_id, rule_ev, cls_score):
        full_count, needed, strict = rule_ev["full_count"], rule_ev["covered_groups"], rule_ev["strict"]
        if strict:
            return "Covered" if full_count >= needed else "Missing"
        if full_count == 0:
            return "Missing"
        if full_count >= needed:
            if requirement_id in {"REQ-009", "REQ-010", "REQ-011"}:
                return "Covered"
            if self.classifier is None:
                return "Covered"
            if OPP_CATEGORY_MAP.get(requirement_id) and cls_score >= self.min_score:
                return "Covered"
            return "Partial"
        return "Partial"


def find_policy_file(name):
    for cand in (RAW / f"{name}_ar.txt", RAW / f"{name}_en.txt", RAW / f"{name}.txt"):
        if cand.exists():
            return cand
    return None


def macro_f1(gold, pred):
    scores = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores)


def run_mode(analyzer, policies, gold):
    """policies: {name: text}. Returns rows [{policy, requirement_id, gold, pred}]."""
    rows = []
    for name, text in policies.items():
        for r in analyzer.analyze(text):
            g = gold.get((name, r["requirement_id"]))
            if g:
                rows.append({"policy": name, "requirement_id": r["requirement_id"], "gold": g, "pred": r["status"]})
    return rows


def summarize(rows):
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    n = len(rows)
    return {"n": n,
            "accuracy": sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0,
            "macro_f1": macro_f1(gold, pred) if n else 0.0,
            "over_coverage": sum(p == "Covered" and g != "Covered" for g, p in zip(gold, pred)),
            "under_coverage": sum(g == "Covered" and p != "Covered" for g, p in zip(gold, pred))}


def per_requirement(rows):
    by = {}
    for r in rows:
        by.setdefault(r["requirement_id"], []).append(r["gold"] == r["pred"])
    return {k: sum(v) / len(v) for k, v in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-scores", type=float, nargs="+", default=[0.5])
    ap.add_argument("--model-source", default=None, help="local folder or Hugging Face repo id (default: local "
                    "models/xlm_roberta_best if it exists, else the default repo of classifier_v2_1.py)")
    ap.add_argument("--no-classifier", action="store_true")
    args = ap.parse_args()

    reqs = load_requirements(CHECKLIST)
    gold = gs.load_gold(GOLD)
    policies = {}
    for name in sorted({n for n, _ in gold}):
        f = find_policy_file(name)
        if f:
            policies[name] = f.read_text(encoding="utf-8", errors="replace")
        else:
            print(f"! no raw text for '{name}' (skipped)")
    print(f"policies: {len(policies)} | gold decisions: {sum(1 for (n, _) in gold if n in policies)}\n")

    modes = {"rules_only": ThresholdAnalyzer(reqs, None)}
    if not args.no_classifier:
        try:
            Adapter, default_model = gs.import_classifier()
            source = gs.resolve_classifier_source(MODEL_DIR, args.model_source, default_model)
            if not source:
                raise FileNotFoundError("no local model folder and no default Hugging Face repo")
            t0 = time.perf_counter()
            base = Adapter(source)
            print(f"classifier '{source}' loaded in {time.perf_counter() - t0:.1f}s")
            if gs.apply_default_labels_if_generic(base):
                print("! model labels were LABEL_0..LABEL_11: ASSUMED the standard OPP-115 order. Verify it!")
            bad = gs.label_name_problems(base, OPP_CATEGORY_MAP)
            if bad:
                print("!! WARNING: model label names do not match rules_v2.OPP_CATEGORY_MAP; missing:", bad)
                print("   every classifier score will be 0 and Covered will silently become Partial.\n")
            clf = gs.MemoClassifier(base)       # one forward pass per paragraph, shared by all modes
            for sc in args.min_scores:
                modes[f"rules+xlmr@{sc:g}"] = ThresholdAnalyzer(reqs, clf, sc)
        except Exception as exc:
            print(f"(classifier unavailable: {type(exc).__name__}: {exc}). Only the rules-only mode is evaluated.\n")

    results, all_rows = {}, []
    for label, analyzer in modes.items():
        t0 = time.perf_counter()
        rows = run_mode(analyzer, policies, gold)
        results[label] = rows
        print(f"[{label}] done in {time.perf_counter() - t0:.1f}s")
        for r in rows:
            all_rows.append({**r, "mode": label})

    print("\n" + "=" * 86)
    print(f"{'mode':18s} {'n':>4s} {'accuracy':>9s} {'macroF1':>8s} {'over-cov':>9s} {'under-cov':>10s}")
    for label, rows in results.items():
        s = summarize(rows)
        print(f"{label:18s} {s['n']:4d} {s['accuracy']*100:8.1f}% {s['macro_f1']:8.3f} {s['over_coverage']:9d} {s['under_coverage']:10d}")
    print("over-cov = predicted Covered but Gold says Partial/Missing (false reassurance)")

    print("\nper-requirement accuracy (%):")
    pr = {label: per_requirement(rows) for label, rows in results.items()}
    labels = list(results)
    print(f"{'req':8s}" + "".join(f"{l[:16]:>18s}" for l in labels))
    for rid in sorted(next(iter(pr.values()))):
        print(f"{rid:8s}" + "".join(f"{pr[l].get(rid, 0)*100:18.0f}" for l in labels))

    out = ROOT / "reports" / "compare_rules_vs_classifier.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "policy", "requirement_id", "gold", "pred"])
        w.writeheader()
        w.writerows(all_rows)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
