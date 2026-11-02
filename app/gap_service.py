"""Service layer between the Streamlit app and the team's GapAnalyzerV2 (src/gap_analyzer/).

Kept free of Streamlit so it can be unit-tested. It does NOT change the analyzer's decisions; it adds
  1. MemoClassifier  - the analyzer calls predict_all_scores(paragraph) once per (requirement x paragraph);
                       caching by text turns ~900 forward passes into ~60 for a 60-paragraph policy.
  2. article evidence - for each requirement, the article text retrieved by the team's RAG pipeline
                       (matched on document AND article number, because numbers repeat across documents).
  3. gold lookup      - to compare with the human-reviewed Gold when rehearsing on the 10 known policies.
"""
import csv
import json
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


# ----------------------------------------------------------------------------- classifier source and labels
OPP115_DEFAULT_LABELS = {   # standard OPP-115 / Hugging Face order (ASSUMED when a model only exposes LABEL_0..LABEL_11)
    0: "Data Retention", 1: "Data Security", 2: "Do Not Track", 3: "First Party Collection/Use",
    4: "International and Specific Audiences", 5: "Introductory/Generic", 6: "Policy Change", 7: "Other",
    8: "Privacy contact information", 9: "Third Party Sharing/Collection", 10: "User Access, Edit and Deletion",
    11: "User Choice/Control"}


def import_classifier():
    """(XLMRClassifierAdapter, DEFAULT_MODEL). Prefers classifier_v2_1.py (can load from Hugging Face),
    falls back to classifier_v2.py (local folder only; DEFAULT_MODEL is then None)."""
    try:
        import classifier_v2_1 as module
    except ImportError:
        import classifier_v2 as module
    return module.XLMRClassifierAdapter, getattr(module, "DEFAULT_MODEL", None)


def resolve_classifier_source(local_dir, env_value=None, default_model=None):
    """Priority: CLASSIFIER_SOURCE env var > local folder (if it exists) > default Hugging Face repo."""
    if env_value:
        return env_value
    if Path(local_dir).exists():
        return str(local_dir)
    return default_model


def apply_default_labels_if_generic(classifier):
    """If the model only exposes LABEL_0..LABEL_11, assume the standard OPP-115 order. Returns True if applied.
    Without this, rules_v2.OPP_CATEGORY_MAP never matches, every score is 0 and Covered silently becomes Partial."""
    names = list(getattr(classifier, "id2label", {}).values())
    if len(names) == 12 and all(str(n).startswith("LABEL_") for n in names):
        classifier.id2label = dict(OPP115_DEFAULT_LABELS)
        if hasattr(classifier, "thresholds"):
            classifier.thresholds = {label: 0.5 for label in classifier.id2label.values()}
        return True
    return False


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


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def make_cached_retriever(pipe, final_k=5, cache_path=None):
    """Memoised RAGPipeline.retrieve(query)['final'], optionally persisted to disk.

    The 15 checklist questions never change, so their retrieval is computed ONCE (slow: embedding + reranker
    on CPU) and saved to cache_path. Next start-ups load it in milliseconds. Delete the file if the legal
    corpus or the index changes."""
    path = Path(cache_path) if cache_path else None
    store, lock = {}, threading.Lock()
    if path and path.exists():
        try:
            store.update(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            pass

    def retrieve(query):
        key = f"{final_k}|{query}"
        with lock:
            if key in store:
                return store[key]
        res = pipe.retrieve(query, final_k=final_k)
        docs = list(res["final"]) if isinstance(res, dict) else list(res)
        with lock:
            store[key] = docs
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(store, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return docs
    retrieve.cache = store
    return retrieve


def warm_up(retriever, requirements, log=print):
    """Run the checklist questions through the retriever; returns the failures. Logs timing per question."""
    failed = []
    for r in requirements:
        t0 = time.perf_counter()
        try:
            retriever(r.verification_question)
            log(f"[warm-up] {r.requirement_id} {time.perf_counter() - t0:5.1f}s")
        except Exception as exc:
            failed.append((r.requirement_id, type(exc).__name__))
            log(f"[warm-up] {r.requirement_id} FAILED {type(exc).__name__}")
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