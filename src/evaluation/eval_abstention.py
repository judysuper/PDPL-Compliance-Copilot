"""Evaluation of citation checking and abstention on the team's gold Q&A set.

Steps (all used from the day-3 notebook):
  1. load_gold          -> read the gold set into standard columns
  2. split_calibration  -> calibration half (tune threshold) / evaluation half (report)
  3. collect_retrieval  -> run retrieval + rerank once per question (no LLM)
  4. calibrate          -> choose the rerank-score threshold on the calibration half
  5. run_eval           -> full pipeline on the evaluation half, one row per question
  6. summarize / save_run / e5_grid / update_readme
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from generation.abstention import calibrate_threshold
from generation.citation_checker import normalize_article_number, to_ascii_digits

QUESTION_COLS = ("question", "Question", "السؤال", "query")
GOLD_COLS = ("gold_articles", "gold_article", "gold_article_numbers", "article_numbers",
             "articles", "article_number", "gold")
ANSWERABLE_COLS = ("answerable_from_sources", "answerable", "is_answerable",
                   "expected_behavior", "expected", "expect")
LANGUAGE_COLS = ("language", "lang", "اللغة")
ID_COLS = ("question_id", "qid", "id")
SOURCE_DOC_COLS = ("source_document", "gold_source", "document")

UNANSWERABLE_VALUES = {
    "no", "false", "0", "n", "unanswerable", "not_covered", "not covered", "notcovered",
    "abstain", "out_of_scope", "out of scope", "غير مغطى", "لا", "غير قابل للإجابة",
}


def _pick(df, given, options):
    if given:
        return given
    return next((c for c in options if c in df.columns), None)


def _parse_articles(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    # "1.0" (pandas float) -> "1";  "5, 8" / "5|8" -> ["5", "8"]
    return [normalize_article_number(n)
            for n in re.findall(r"\d+(?:\.0+)?(?!\d)", to_ascii_digits(str(value)))]


def _doc_prefix(value):
    """'pdpl_ar' / 'implementing_en' / 'transfer_ar' -> 'pdpl' / 'implementing' / 'transfer'."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return ""
    return str(value).strip().lower().split("_")[0]


def _parse_answerable(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip().lower() not in UNANSWERABLE_VALUES


def load_gold(path, question_col=None, gold_col=None, answerable_col=None, language_col=None,
              id_col=None, source_col=None):
    """Read the gold set (csv / xlsx / json) into:
    qid, question, gold_articles, gold_source, answerable, language."""
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    elif path.suffix == ".json":
        raw = pd.read_json(path)
    else:
        raw = pd.read_csv(path, encoding="utf-8-sig")

    q_col = _pick(raw, question_col, QUESTION_COLS)
    if q_col is None:
        raise ValueError(f"No question column found. Columns: {list(raw.columns)}")
    g_col = _pick(raw, gold_col, GOLD_COLS)
    a_col = _pick(raw, answerable_col, ANSWERABLE_COLS)
    l_col = _pick(raw, language_col, LANGUAGE_COLS)
    i_col = _pick(raw, id_col, ID_COLS)
    s_col = _pick(raw, source_col, SOURCE_DOC_COLS)

    df = pd.DataFrame({
        "qid": raw[i_col].astype(str) if i_col else [f"q{i + 1:03d}" for i in range(len(raw))],
        "question": raw[q_col].astype(str).str.strip(),
    })
    df["gold_articles"] = raw[g_col].apply(_parse_articles) if g_col else [[] for _ in range(len(raw))]
    # the same article number exists in PDPL, the Implementing and the Transfer regulation,
    # so the gold document is kept to match (document, article) and not the number alone
    df["gold_source"] = raw[s_col].apply(_doc_prefix) if s_col else ""

    if a_col:
        df["answerable"] = raw[a_col].apply(_parse_answerable)
    elif g_col:
        df["answerable"] = df["gold_articles"].apply(bool)  # no gold article -> not covered
    else:
        raise ValueError("Need an 'answerable' column or a gold-article column to know "
                         "which questions are unanswerable.")

    if l_col:
        df["language"] = raw[l_col].astype(str).str.lower().str[:2]
    else:
        df["language"] = df["question"].apply(
            lambda q: "ar" if re.search(r"[؀-ۿ]", q) else "en")

    bad = df[df["answerable"] & (df["gold_articles"].apply(len) == 0)]
    if len(bad):
        print(f"⚠️ {len(bad)} answerable questions have no gold article: {bad['qid'].tolist()}")

    print(f"columns used -> id: {i_col!r}, question: {q_col!r}, gold: {g_col!r}, "
          f"source: {s_col!r}, answerable: {a_col!r}, language: {l_col!r}")
    print(df.groupby(["language", "answerable"]).size().rename("n").to_string())
    return df


def split_calibration(df, calib_frac=0.5, seed=42):
    """Stratified split by (answerable, language): tune on one half, report on the other."""
    calib = df.groupby(["answerable", "language"]).sample(frac=calib_frac, random_state=seed)
    evaluation = df.drop(calib.index)
    return calib.reset_index(drop=True), evaluation.reset_index(drop=True)


def collect_retrieval(pipeline, df, **retrieve_kwargs):
    """Run retrieval + rerank once per question. Returns {qid: retrieved}."""
    cache = {}
    for row in df.itertuples():
        cache[row.qid] = pipeline.retrieve(row.question, **retrieve_kwargs)
    return cache


def calibrate(df, cache, max_false_refusal=0.10):
    scores = [cache[q]["max_rerank_score"] for q in df["qid"]]
    return calibrate_threshold(scores, df["answerable"].tolist(), max_false_refusal)


def save_threshold(calibration, path, extra=None):
    """Write the chosen threshold to configs/abstention.json (read by RAGPipeline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in calibration.items() if k != "curve"}
    payload["calibrated_at"] = datetime.now().isoformat(timespec="seconds")
    payload.update(extra or {})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _doc_source(doc):
    """'pdpl' / 'implementing' / 'transfer' for a retrieved article (dense and BM25 results)."""
    source = doc.get("source") or ""
    if not source and doc.get("source_file"):
        source = str(doc["source_file"]).split("_")[0]
    return source


def _is_gold(doc, row):
    if normalize_article_number(doc.get("article_number", "")) not in row.gold_articles:
        return False
    return not row.gold_source or _doc_source(doc) == row.gold_source


def gold_rank(retrieved_docs, row):
    """1-based rank of the gold (document, article) in a result list, or None."""
    for i, d in enumerate(retrieved_docs, start=1):
        if _is_gold(d, row):
            return i
    return None


def run_eval(pipeline, df, cache, provider=None, label="full", max_consecutive_errors=3,
             **ask_kwargs):
    """Full pipeline on every question. One row per question, with blank human-review columns.

    Stops early after `max_consecutive_errors` failed questions in a row (e.g. the LLM
    service is down), instead of filling the results with errors.
    """
    rows = []
    consecutive_errors = 0
    for row in df.itertuples():
        expected = "answer" if row.answerable else "abstain"
        base = {"config": label, "qid": row.qid, "language": row.language,
                "question": row.question, "expected": expected,
                "gold_articles": " | ".join(row.gold_articles)}
        try:
            r = pipeline.ask(row.question, provider=provider, retrieved=cache[row.qid], **ask_kwargs)
        except Exception as exc:  # keep going, but record the failure
            rows.append({**base, "error": repr(exc)})
            consecutive_errors += 1
            print(f"[{row.qid}] error: {str(exc)[:150]}")
            if consecutive_errors >= max_consecutive_errors:
                print(f"Stopping: {consecutive_errors} questions failed in a row. "
                      "Partial results are returned; re-run when the LLM service is available.")
                break
            continue
        consecutive_errors = 0

        check = r["citation_check"] or {}
        cited = check.get("cited_articles") or []
        final = r["final_sources"]
        # cites gold = one of the [Source n] tags points to the gold (document, article)
        cites_gold = None
        if not r["abstained"] and row.gold_articles:
            cites_gold = any(1 <= n <= len(final) and _is_gold(final[n - 1], row)
                             for n in check.get("cited_sources") or [])
        rows.append({
            **base,
            "max_rerank_score": r["max_rerank_score"],
            "threshold": r["threshold"],
            "gold_source": row.gold_source,
            "gold_rank_reranked": gold_rank(cache[row.qid]["final"], row) if row.gold_articles else None,
            "final_articles": " | ".join(f"{d['source']}:{d['article_number']}" for d in final),
            "llm_model": r.get("llm_model"),
            "prompt_style": r.get("prompt_style"),
            "abstained": r["abstained"],
            "abstention_reason": r["abstention_reason"],
            "abstain_ok": r["abstained"] == (expected == "abstain"),
            "first_attempt_valid": r["first_attempt_valid"],
            "n_attempts": r["n_attempts"],
            "citation_valid": check.get("valid") if not r["abstained"] else None,
            "cited_articles": " | ".join(cited),
            "unsupported_articles": " | ".join(check.get("unsupported_articles") or []),
            "cross_referenced_articles": " | ".join(check.get("cross_referenced_articles") or []),
            "mismatched_pairs": json.dumps(check.get("mismatched_pairs") or [], ensure_ascii=False),
            "cites_gold_article": cites_gold,
            "min_claim_overlap": check.get("min_claim_overlap"),
            "needs_human_review": not r["abstained"],
            "review_supported_grader1": "",   # yes / partial / no
            "review_supported_grader2": "",
            "review_notes": "",
            "answer": r["answer"],
            "raw_answer": r["raw_answer"],
            "total_s": round(r["timings"]["total_s"], 2),
            "error": "",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------- resumable runs (checkpoints)
BOOL_COLS = ("abstained", "abstain_ok", "first_attempt_valid", "citation_valid",
             "cites_gold_article", "needs_human_review")
NUM_COLS = ("max_rerank_score", "threshold", "gold_rank_reranked", "min_claim_overlap",
            "total_s", "n_attempts")


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return {"true": True, "false": False}.get(str(value).strip().lower())


def restore_types(df):
    """Undo what a CSV round-trip does to run_eval rows ("True" -> True, "" -> None...)."""
    df = df.copy()
    for col in BOOL_COLS:
        if col in df:
            df[col] = df[col].map(_to_bool).astype(object)
    for col in NUM_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "error" in df:
        df["error"] = df["error"].fillna("").astype(str)
    if "abstention_reason" in df:
        df["abstention_reason"] = df["abstention_reason"].map(
            lambda v: None if pd.isna(v) or str(v).strip() in ("", "None", "nan") else v)
    for col in ("qid", "config"):
        if col in df:
            df[col] = df[col].astype(str)
    return df


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return restore_types(pd.read_csv(path, encoding="utf-8-sig"))


def run_eval_resumable(pipeline, df, cache, checkpoint, provider=None, label="full",
                       stop_on_error=True, verbose=True, **ask_kwargs):
    """run_eval one question at a time, saving every answer to `checkpoint` (CSV).

    Re-running the same call skips the (label, qid) pairs already saved, so a Colab
    disconnect costs at most one question. Rows that ended in an error are retried.
    The checkpoint may hold several labels (styles, E5 configs); each keeps its rows.
    Returns this label's rows.
    """
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = load_checkpoint(checkpoint)
    if len(saved):
        saved = saved[saved["error"] == ""]
    done = set(zip(saved["config"], saved["qid"])) if len(saved) else set()
    todo = [i for i in range(len(df)) if (label, str(df.iloc[i]["qid"])) not in done]
    if verbose:
        print(f"[{label}] {len(df) - len(todo)}/{len(df)} already saved, {len(todo)} to run", flush=True)

    parts = [saved] if len(saved) else []
    for n, i in enumerate(todo, start=1):
        one = df.iloc[[i]]
        qid = str(one.iloc[0]["qid"])
        started = time.monotonic()
        res = run_eval(pipeline, one, cache, provider=provider, label=label,
                       max_consecutive_errors=1, **ask_kwargs)
        error = str(res.iloc[0].get("error", "") or "")
        if error and error != "nan":
            if stop_on_error:
                raise RuntimeError(
                    f"Stopped at {label} / {qid}: {error[:300]}\n"
                    f"Answers so far are saved in {checkpoint}; run the cell again to continue.")
            continue
        parts.append(res)
        pd.concat(parts, ignore_index=True).to_csv(checkpoint, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  [{label}] {qid} done in {time.monotonic() - started:.1f}s ({n}/{len(todo)})",
                  flush=True)

    rows = load_checkpoint(checkpoint)
    return rows[rows["config"] == label].reset_index(drop=True) if len(rows) else rows


def _rate(series):
    series = series.dropna()
    return round(float(series.mean()), 3) if len(series) else None


def summarize(results):
    """Metrics for the report. Abstention metrics follow the brief (E5 / evaluation table)."""
    ok = results[results["error"].fillna("") == ""] if "error" in results else results
    ans = ok[ok["expected"] == "answer"]
    una = ok[ok["expected"] == "abstain"]
    answered = ok[~ok["abstained"].astype(bool)]
    generated = ok[ok["first_attempt_valid"].notna()]

    summary = {
        "n_questions": int(len(ok)),
        "n_errors": int(len(results) - len(ok)),
        "n_answerable": int(len(ans)),
        "n_unanswerable": int(len(una)),
        "abstention_accuracy": _rate(una["abstained"].astype(float)),
        "false_refusal_rate": _rate(ans["abstained"].astype(float)),
        "n_answered": int(len(answered)),
        "first_attempt_citation_valid_rate": _rate(generated["first_attempt_valid"].astype(float)),
        "repaired_by_retry": int(((generated["first_attempt_valid"] == False)  # noqa: E712
                                  & (~generated["abstained"].astype(bool))).sum()),
        "cites_gold_article_rate": _rate(answered["cites_gold_article"].dropna().astype(float)),
        "gold_in_final_sources_rate": _rate(ans["gold_rank_reranked"].notna().astype(float))
        if "gold_rank_reranked" in ans else None,
        "abstention_reasons": ok["abstention_reason"].dropna().value_counts().to_dict(),
        "llm_models": ok["llm_model"].dropna().value_counts().to_dict() if "llm_model" in ok else {},
    }
    by_lang = {}
    for lang, g in ok.groupby("language"):
        by_lang[lang] = {
            "abstention_accuracy": _rate(g[g["expected"] == "abstain"]["abstained"].astype(float)),
            "false_refusal_rate": _rate(g[g["expected"] == "answer"]["abstained"].astype(float)),
        }
    summary["by_language"] = by_lang
    return summary


def save_run(results, summary, out_dir, name="citation_abstention"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{name}_{stamp}.csv"
    json_path = out_dir / f"{name}_{stamp}_summary.json"
    # lowest overlap first: the rows a human should review first
    results.sort_values("min_claim_overlap", na_position="first").to_csv(
        csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


# --------------------------------------------------- phase 1b: prompt style choice
def choose_prompt_style(table, max_false_refusal=0.10):
    """Pick a style from compare_prompt_styles' table.

    Rule: highest abstention accuracy among styles whose false-refusal rate is within
    the budget; ties -> lower false-refusal rate; then more citations valid on the first
    attempt; then the earlier (simpler) style in the table. If no style is within the
    budget, the one with the lowest false-refusal rate.

    (2026-10-26: the false-refusal tie-break was added before running system_v2. The
    first version skipped it, so two styles with equal abstention accuracy were split
    on citations even when one refused more answerable questions. It does not change
    the choice among the first three styles.)
    """
    rows = table.reset_index(drop=True)
    ok = rows[rows["false_refusal_rate"].fillna(1) <= max_false_refusal]
    if ok.empty:
        return rows.sort_values(["false_refusal_rate"], kind="stable").iloc[0]["prompt_style"]
    ranked = ok.assign(
        _abst=ok["abstention_accuracy"].fillna(0),
        _fr=ok["false_refusal_rate"].fillna(1),
        _cite=ok["first_attempt_citation_valid_rate"].fillna(0),
        _order=ok.index,
    ).sort_values(["_abst", "_fr", "_cite", "_order"], ascending=[False, True, False, True],
                  kind="stable")
    return ranked.iloc[0]["prompt_style"]


def style_table(rows, styles):
    """One summary row per prompt style from run_eval rows labelled "style_<name>"."""
    out = []
    for style in styles:
        res = rows[rows["config"] == f"style_{style}"] if len(rows) else rows
        if not len(res):
            continue
        s = summarize(res)
        out.append({
            "prompt_style": style,
            "n_questions": s["n_questions"],
            "abstention_accuracy": s["abstention_accuracy"],
            "false_refusal_rate": s["false_refusal_rate"],
            "first_attempt_citation_valid_rate": s["first_attempt_citation_valid_rate"],
            "cites_gold_article_rate": s["cites_gold_article_rate"],
            "n_errors": s["n_errors"],
            "reasons": s["abstention_reasons"],
        })
    return pd.DataFrame(out)


def compare_prompt_styles(pipeline, df, cache, provider,
                          styles=("single", "system", "system_examples", "system_v2"),
                          max_false_refusal=0.10, checkpoint=None):
    """Run each prompt style on the calibration split with the threshold OFF.

    The threshold is off so the table shows the LLM's own abstention for each style.
    With `checkpoint`, answers are saved one by one and styles already in the file are
    not run again (e.g. the first three styles from an earlier session).
    Returns (table, rows, chosen_style).
    """
    parts = []
    for style in styles:
        kwargs = dict(provider=provider, label=f"style_{style}", prompt_style=style,
                      min_rerank_score=None)
        if checkpoint:
            parts.append(run_eval_resumable(pipeline, df, cache, checkpoint, **kwargs))
        else:
            parts.append(run_eval(pipeline, df, cache, **kwargs))
    rows = pd.concat([p for p in parts if len(p)], ignore_index=True)
    table = style_table(rows, styles)
    incomplete = table[table["n_questions"] + table["n_errors"] < len(df)]
    if len(incomplete):
        raise RuntimeError(f"Some styles do not cover all {len(df)} questions: "
                           f"{incomplete['prompt_style'].tolist()}")
    return table, rows, choose_prompt_style(table, max_false_refusal)


def save_generation_config(prompt_style, path, extra=None):
    """Write the chosen prompt style to configs/generation.json (read by RAGPipeline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"prompt_style": prompt_style,
               "chosen_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(extra or {})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def e5_grid(pipeline, df, cache, provider, threshold, checkpoint=None):
    """Experiment E5: abstain instruction on/off x retrieval threshold on/off.

    Uses the pipeline's prompt style. With `checkpoint`, answers are saved one by one.
    """
    configs = [
        ("no_instruction_no_threshold", False, None),
        ("instruction_only", True, None),
        ("threshold_only", False, threshold),
        ("instruction_and_threshold", True, threshold),
    ]
    tables, rows = [], []
    for label, instruction, t in configs:
        if t is None and "threshold" in label and not label.startswith("no_"):
            continue  # no calibrated threshold: skip the threshold configs
        kwargs = dict(provider=provider, label=label, abstain_instruction=instruction,
                      min_rerank_score=t)
        if checkpoint:
            res = run_eval_resumable(pipeline, df, cache, checkpoint, **kwargs)
        else:
            res = run_eval(pipeline, df, cache, **kwargs)
        s = summarize(res)
        rows.append({"config": label, "abstain_instruction": instruction, "threshold": t,
                     "abstention_accuracy": s["abstention_accuracy"],
                     "false_refusal_rate": s["false_refusal_rate"],
                     "first_attempt_citation_valid_rate": s["first_attempt_citation_valid_rate"],
                     "reasons": s["abstention_reasons"]})
        tables.append(res)
    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True)


# ------------------------------------------------------------------- README
README_START = "<!-- DAY3-CITATION-ABSTENTION:START -->"
README_END = "<!-- DAY3-CITATION-ABSTENTION:END -->"


def readme_section(summary, calibration, provider, results_csv, e5_table=None,
                   style_table=None, prompt_style=None):
    t = calibration.get("min_rerank_score")
    lines = [
        README_START,
        "## Citation check and abstention",
        "",
        "**Files:** `src/generation/citation_checker.py`, `src/generation/abstention.py`, "
        "`src/rag_pipeline.py`, `src/evaluation/eval_abstention.py`, "
        "`tests/test_citation_abstention.py`.",
        "",
        "**How it works**",
        "- Before the LLM: no retrieved articles, or best rerank score below the calibrated "
        "threshold -> the system refuses.",
        "- After the LLM: `NOT_COVERED:` reply -> refuses. Otherwise the citation checker "
        "verifies that every `[Source n]` was sent to the model, every article number cited "
        "was retrieved, and every `[Source n] (Article x)` pair matches. Failed answers are "
        "regenerated once with the allowed citations; if they still fail, the system refuses.",
        "",
        "**Run**",
        "```bash",
        "python -m pytest tests/ -q",
        'python src/rag_pipeline.py "ما حقوق صاحب البيانات الشخصية؟" --provider gemini',
        "```",
        "",
        f"**Threshold:** min rerank score = `{'off' if t is None else round(t, 4)}` (chosen on the calibration half: "
        f"{calibration['n_answerable']} answerable / {calibration['n_unanswerable']} unanswerable "
        f"questions, max false-refusal {calibration['max_false_refusal']:.0%}). Stored in "
        "`configs/abstention.json`.",
        "",
        f"**LLM:** provider `{provider}`, model(s) {summary.get('llm_models') or '-'}, "
        f"prompt style `{prompt_style or '-'}` (chosen on the calibration half, "
        "`configs/generation.json`).",
        "",
        f"**Results on the evaluation half** (file: `{results_csv}`)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions (answerable / unanswerable) | {summary['n_answerable']} / {summary['n_unanswerable']} |",
        f"| Abstention accuracy on unanswerable | {summary['abstention_accuracy']} |",
        f"| False-refusal rate on answerable | {summary['false_refusal_rate']} |",
        f"| Citations valid on first attempt | {summary['first_attempt_citation_valid_rate']} |",
        f"| Answers repaired by the retry | {summary['repaired_by_retry']} |",
        f"| Answered questions citing a gold article | {summary['cites_gold_article_rate']} |",
        f"| Gold article among the sources sent to the LLM (answerable) | {summary.get('gold_in_final_sources_rate')} |",
        f"| Abstention reasons | {summary['abstention_reasons']} |",
        f"| LLM model(s) that answered | {summary.get('llm_models')} |",
    ]
    if style_table is not None and len(style_table):
        lines += ["", "**Prompt style (calibration half, threshold off)**", "",
                  "| Style | Abstention accuracy | False-refusal rate | Citations valid on 1st attempt |",
                  "|---|---|---|---|"]
        for r in style_table.itertuples():
            lines.append(f"| {r.prompt_style} | {r.abstention_accuracy} | {r.false_refusal_rate} | "
                         f"{r.first_attempt_citation_valid_rate} |")
    if e5_table is not None and len(e5_table):
        lines += ["", "**Experiment E5**", "",
                  "| Config | Abstention accuracy | False-refusal rate |", "|---|---|---|"]
        for r in e5_table.itertuples():
            lines.append(f"| {r.config} | {r.abstention_accuracy} | {r.false_refusal_rate} |")
    lines += [
        "",
        "**Limits (what is NOT guaranteed)**",
        "- The checker proves a cited article was retrieved, not that it supports the claim. "
        "Support is judged by two team members (`review_supported_*` columns in the results CSV, "
        "sorted so the lowest lexical-overlap answers come first).",
        "- Article numbers repeat across the PDPL, the Implementing Regulation and the Transfer "
        "Regulation; an article mentioned without its `[Source n]` tag is accepted if any "
        "retrieved document has that number.",
        "- A sentence such as \"Article 999 is not in the sources\" counts as an unsupported citation.",
        "- An article that is only cross-referenced inside a retrieved text (e.g. Article 6 refers to "
        "Article 5) is accepted with a warning, because its own text was not retrieved.",
        "- The threshold was tuned on a small calibration set; the numbers above come from the "
        "other half and will move with more questions.",
        README_END,
    ]
    return "\n".join(lines)


def update_readme(readme_path, section):
    """Insert or replace the day-3 section without touching the rest of README.md."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if README_START in text and README_END in text:
        before = text.split(README_START)[0]
        after = text.split(README_END, 1)[1]
        text = before + section + after
    else:
        text = text.rstrip() + ("\n\n" if text else "") + section + "\n"
    path.write_text(text, encoding="utf-8")
    return path