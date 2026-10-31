"""Service layer between the Streamlit app and the team's GapAnalyzerV2 (src/gap_analyzer/).

Kept free of Streamlit so it can be unit-tested. It does NOT change the analyzer's decisions; it adds
  1. MemoClassifier  - the analyzer calls predict_all_scores(paragraph) once per (requirement x paragraph);
                       caching by text turns ~900 forward passes into ~60 for a 60-paragraph policy.
  2. article evidence - for each requirement, the article text retrieved by the team's RAG pipeline
                       (matched on document AND article number, because numbers repeat across documents).
  3. gold lookup      - to compare with the human-reviewed Gold when rehearsing on the 10 known policies.
"""
import csv
import re
import threading
import time
from pathlib import Path

STATUSES = ("Covered", "Partial", "Missing")
POLICY_NAME_MAP = {"P01": "stc", "P02": "alrajhi", "P03": "jarir", "P04": "hungerstation", "P05": "tawakkalna",
                   "P06": "absher", "P07": "mobily", "P08": "bankalbilad", "P09": "tamara", "P10": "nahdi"}


class MemoClassifier:
    """Wraps XLMRClassifierAdapter: same predict_all_scores(text) API, cached by exact text."""

    def __init__(self, base):
        self.base, self.cache, self.lock, self.calls = base, {}, threading.Lock(), 0
        self.id2label = getattr(base, "id2label", {})

    def predict_all_scores(self, text):
        with self.lock:
            if text in self.cache:
                return self.cache[text]
        scores = self.base.predict_all_scores(text)
        with self.lock:
            self.cache[text] = scores
            self.calls += 1
        return scores


def label_name_problems(classifier, opp_category_map):
    """Category names that rules_v2.OPP_CATEGORY_MAP expects but the model does not expose (e.g. the model
    saved LABEL_0..LABEL_11). Then every classifier score is 0 and Covered silently becomes Partial."""
    needed = {name for names in opp_category_map.values() for name in names}
    have = set(getattr(classifier, "id2label", {}).values())
    return sorted(needed - have)


# ----------------------------------------------------------------------------- article evidence
def source_prefix(source_document):
    """Checklist source_document -> corpus prefix ('pdpl' / 'implementing' / 'transfer')."""
    t = str(source_document)
    if "اللائحة التنفيذية" in t or "implementing" in t.lower():
        return "implementing"
    if "نقل" in t or "transfer" in t.lower():
        return "transfer"
    return "pdpl"


def _num(value):
    m = re.search(r"\d+", str(value))
    return m.group(0) if m else str(value).strip()


def _get(doc, *names):
    for n in names:
        v = doc.get(n) if isinstance(doc, dict) else getattr(doc, n, None)
        if v not in (None, ""):
            return v
    return None


def pick_article(docs, source_document, article_number):
    want_p, want_n = source_prefix(source_document), _num(article_number)
    for d in docs or []:
        prefix = str(_get(d, "source", "source_document") or "").lower().split("_")[0]
        if _num(_get(d, "article_number", "article") or "") == want_n and prefix == want_p:
            return d
    return None


def make_cached_retriever(pipe, final_k=5):
    """Memoised RAGPipeline.retrieve(query)['final']; checklist questions never change, so warm it once."""
    store, lock = {}, threading.Lock()

    def retrieve(query):
        with lock:
            if query in store:
                return store[query]
        res = pipe.retrieve(query, final_k=final_k)
        docs = list(res["final"]) if isinstance(res, dict) else list(res)
        with lock:
            store[query] = docs
        return docs
    retrieve.cache = store
    return retrieve


def warm_up(retriever, requirements):
    failed = []
    for r in requirements:
        try:
            retriever(r.verification_question)
        except Exception as exc:
            failed.append((r.requirement_id, type(exc).__name__))
    return failed


# ----------------------------------------------------------------------------- analysis
def run_gap(analyzer, policy_text, retriever=None):
    """analyzer.analyze(text) + article evidence. Returns {results, summary, seconds, warnings}."""
    warnings = []
    t0 = time.perf_counter()
    results = analyzer.analyze(policy_text)
    seconds = round(time.perf_counter() - t0, 2)
    if not str(policy_text).strip() or len(str(policy_text)) < 300:
        warnings.append("النص قصير جدًا أو فارغ: تأكد أنه السياسة كاملة.")
    for r in results:
        r["retrieved_article_text"], r["article_retrieved"], r["top_retrieved"] = None, None, []
        if retriever is None:
            continue
        try:
            docs = retriever(r["verification_question"])
        except Exception as exc:
            warnings.append(f"الاسترجاع فشل في {r['requirement_id']} ({type(exc).__name__})")
            continue
        hit = pick_article(docs, r["source_document"], r["article_number"])
        r["article_retrieved"] = hit is not None
        r["retrieved_article_text"] = _get(hit, "article_text", "text") if hit else None
        r["top_retrieved"] = [f'{_get(d, "source")} / {_get(d, "article_number")}' for d in docs[:3]]
    summary = {s: sum(r["status"] == s for r in results) for s in STATUSES}
    return {"results": results, "summary": summary, "seconds": seconds, "warnings": warnings}


# ----------------------------------------------------------------------------- gold (for rehearsal only)
def load_gold(path):
    """{(policy_name, requirement_id): status} from pdpl_local_policies_gold.csv (policy_id P01..P10)."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = POLICY_NAME_MAP.get(row["policy_id"].strip())
            if name:
                out[(name, row["requirement_id"].strip())] = row["gold_status"].strip()
    return out


def policy_name_from_stem(stem):
    return re.sub(r"_(ar|en)$", "", stem.lower())
