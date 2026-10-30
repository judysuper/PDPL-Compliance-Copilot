from __future__ import annotations
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]

GOLD = ROOT / "data" / "local_policies" / "gold" / "pdpl_local_policies_gold.csv"
V2_ROOT = ROOT / "reports" / "gap_analyzer_v2" / "evaluation"
OUT = ROOT / "reports" / "gap_analyzer_v3_gold_cv"
OUT.mkdir(parents=True, exist_ok=True)

POLICY_NAME_MAP = {
    "P01": "stc",
    "P02": "alrajhi",
    "P03": "jarir",
    "P04": "hungerstation",
    "P05": "tawakkalna",
    "P06": "absher",
    "P07": "mobily",
    "P08": "bankalbilad",
    "P09": "tamara",
    "P10": "nahdi",
}

LABELS = ["Covered", "Partial", "Missing"]

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def count_terms(value: str) -> int:
    value = (value or "").strip()
    if not value:
        return 0
    return len([x for x in value.split(";") if x.strip()])

def build_dataset() -> pd.DataFrame:
    gold_rows = read_csv(GOLD)
    if len(gold_rows) != 150:
        raise ValueError(f"Expected 150 Gold rows; found {len(gold_rows)}")

    gold = {}
    org = {}
    for r in gold_rows:
        pid = r["policy_id"].strip()
        rid = r["requirement_id"].strip()
        status = r["gold_status"].strip()
        if status not in LABELS:
            raise ValueError(f"Invalid Gold status: {status}")
        gold[(pid, rid)] = status
        org[pid] = r["organization"].strip()

    rows = []
    for pid, pname in POLICY_NAME_MAP.items():
        fp = V2_ROOT / pname / f"{pname}_gap_analysis_v2.csv"
        if not fp.exists():
            raise FileNotFoundError(
                f"Missing V2 output: {fp}\n"
                "Run run_all_policies_v2.py first."
            )

        for r in read_csv(fp):
            rid = r["requirement_id"].strip()
            key = (pid, rid)
            if key not in gold:
                raise KeyError(f"Missing Gold row for {key}")

            score = float(r.get("classifier_score") or 0.0)
            evidence = r.get("evidence_text") or ""

            rows.append({
                "policy_id": pid,
                "organization": org[pid],
                "requirement_id": rid,
                "v2_status": (r.get("status") or "").strip(),
                "classifier_category": (r.get("classifier_category") or "NONE").strip() or "NONE",
                "classifier_score": score,
                "matched_terms_count": count_terms(r.get("matched_terms") or ""),
                "evidence_length": len(evidence),
                "gold": gold[key],
            })

    df = pd.DataFrame(rows)
    if len(df) != 150:
        raise ValueError(f"Expected 150 merged rows; found {len(df)}")
    return df

CAT = ["requirement_id", "v2_status", "classifier_category"]
NUM = ["classifier_score", "matched_terms_count", "evidence_length"]

def make_model(C=1.0, class_weight=None):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
    )
    return Pipeline([("prep", prep), ("clf", clf)])

def inner_select(train_df):
    groups = train_df["policy_id"].to_numpy()
    n_splits = min(4, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    candidates = [
        (0.05, None), (0.1, None), (0.5, None), (1.0, None),
        (2.0, None), (10.0, None),
        (0.1, "balanced"), (0.5, "balanced"), (1.0, "balanced"),
        (2.0, "balanced"), (10.0, "balanced"),
    ]

    best = None
    for C, cw in candidates:
        scores = []
        for tr_idx, va_idx in gkf.split(train_df, train_df["gold"], groups):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            model = make_model(C=C, class_weight=cw)
            model.fit(tr[CAT + NUM], tr["gold"])
            pred = model.predict(va[CAT + NUM])
            scores.append(
                f1_score(
                    va["gold"], pred,
                    labels=LABELS, average="macro", zero_division=0
                )
            )
        mean_score = float(np.mean(scores))
        if best is None or mean_score > best["inner_macro_f1"]:
            best = {"inner_macro_f1": mean_score, "C": C, "class_weight": cw}
    return best

def main():
    df = build_dataset()

    print("=" * 78)
    print("GAP ANALYZER V3 — FINAL GOLD NESTED LEAVE-ONE-POLICY-OUT CV")
    print("=" * 78)
    print("Rows:", len(df))
    print("Policies:", df["policy_id"].nunique())
    print("Gold distribution:", df["gold"].value_counts().to_dict())
    print()

    # V2 direct baseline against final Gold
    v2_acc = accuracy_score(df["gold"], df["v2_status"])
    v2_macro = f1_score(
        df["gold"], df["v2_status"],
        labels=LABELS, average="macro", zero_division=0
    )
    print("V2 baseline against FINAL Gold:")
    print(f"Accuracy: {v2_acc*100:.2f}%")
    print(f"Macro-F1: {v2_macro:.4f}")
    print()

    oof = []
    selected = []

    for held_out in sorted(df["policy_id"].unique()):
        train = df[df["policy_id"] != held_out].reset_index(drop=True)
        test = df[df["policy_id"] == held_out].reset_index(drop=True)

        hp = inner_select(train)
        model = make_model(C=hp["C"], class_weight=hp["class_weight"])
        model.fit(train[CAT + NUM], train["gold"])

        pred = model.predict(test[CAT + NUM])
        prob = model.predict_proba(test[CAT + NUM])
        classes = list(model.named_steps["clf"].classes_)

        fold_acc = accuracy_score(test["gold"], pred)
        fold_f1 = f1_score(
            test["gold"], pred,
            labels=LABELS, average="macro", zero_division=0
        )

        selected.append({
            "held_out_policy": held_out,
            "organization": test["organization"].iloc[0],
            "C": hp["C"],
            "class_weight": str(hp["class_weight"]),
            "inner_macro_f1": hp["inner_macro_f1"],
            "fold_accuracy": fold_acc,
            "fold_macro_f1": fold_f1,
        })

        for i, row in test.iterrows():
            rec = row.to_dict()
            rec["v3_gold_prediction"] = pred[i]
            rec["confidence"] = float(np.max(prob[i]))
            for j, cls in enumerate(classes):
                rec[f"prob_{cls}"] = float(prob[i, j])
            oof.append(rec)

        print(
            f"{held_out} {test['organization'].iloc[0]:18s} "
            f"acc={fold_acc*100:6.2f}%  "
            f"macroF1={fold_f1:.3f}  "
            f"C={hp['C']} cw={hp['class_weight']}"
        )

    out = pd.DataFrame(oof).sort_values(["policy_id", "requirement_id"])
    hp_df = pd.DataFrame(selected)

    acc = accuracy_score(out["gold"], out["v3_gold_prediction"])
    macro = f1_score(
        out["gold"], out["v3_gold_prediction"],
        labels=LABELS, average="macro", zero_division=0
    )

    print()
    print("=" * 78)
    print("FINAL GOLD V3 OUT-OF-FOLD RESULT")
    print("=" * 78)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro-F1: {macro:.4f}")
    print()
    print("CLASSIFICATION REPORT")
    print(classification_report(
        out["gold"], out["v3_gold_prediction"],
        labels=LABELS, zero_division=0
    ))
    print("CONFUSION MATRIX [Covered, Partial, Missing]")
    print(confusion_matrix(
        out["gold"], out["v3_gold_prediction"], labels=LABELS
    ))

    print()
    print("PER-POLICY ACCURACY")
    for org, g in out.groupby("organization"):
        a = accuracy_score(g["gold"], g["v3_gold_prediction"])
        print(f"{org:20s} {len(g):3d} decisions  accuracy={a*100:.2f}%")

    print()
    print("PER-REQUIREMENT ACCURACY")
    for rid, g in out.groupby("requirement_id"):
        a = accuracy_score(g["gold"], g["v3_gold_prediction"])
        print(f"{rid:8s} {len(g):3d} decisions  accuracy={a*100:.2f}%")

    out.to_csv(
        OUT / "v3_gold_oof_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    hp_df.to_csv(
        OUT / "v3_gold_selected_hyperparameters.csv",
        index=False, encoding="utf-8-sig"
    )

    summary = {
        "evaluation": "nested leave-one-policy-out cross-validation",
        "labels": "human-reviewed final Gold",
        "n_policies": 10,
        "n_decisions": 150,
        "v2_accuracy": v2_acc,
        "v2_macro_f1": v2_macro,
        "v3_accuracy": acc,
        "v3_macro_f1": macro,
    }
    (OUT / "v3_gold_cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print()
    print("Saved to:", OUT)

if __name__ == "__main__":
    main()
