"""Lazy, cached loaders for the Streamlit app (keeps the page fast to open).

Classifier source (first match wins): env CLASSIFIER_SOURCE > local models/xlm_roberta_best > the default
Hugging Face repo defined in classifier_v2_1.py (downloaded on first use, then cached by transformers).
USE_CLASSIFIER=0 skips the classifier (rules only; fastest for UI work). After the first successful download
set HF_HUB_OFFLINE=1 so the demo never waits for the network.

rag()  -> RAGPipeline + generation module       (loaded on the first question)
gap()  -> GapAnalyzerV2 + classifier + retriever (loaded on the first policy analysis)
preload() loads both (use for the live demo:  PRELOAD=1 streamlit run app/streamlit_app.py)

Retrieval for the 15 checklist questions is saved to cache/checklist_retrieval.json, so only the very first
run pays for it. Stage timings are printed to the terminal.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "gap_analyzer", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CHECKLIST = ROOT / "data" / "pdpl_checklist" / "member1_pdpl_compliance_checklist.csv"
MODEL_DIR = ROOT / "models" / "xlm_roberta_best"
RETRIEVAL_CACHE = ROOT / "cache" / "checklist_retrieval.json"

STATUS = {"rag_loaded": False, "gap_loaded": False, "clf_loaded": False, "clf_warning": None,
          "clf_source": None, "clf_notice": None}


def _log(msg):
    print(msg, flush=True)


@st.cache_resource(show_spinner="تحميل محرك الأسئلة (RAG) لأول مرة... قد يستغرق دقائق")
def rag():
    t0 = time.perf_counter()
    from generation import generation as gen
    from rag_pipeline import RAGPipeline
    pipe = RAGPipeline()
    _log(f"[load] RAGPipeline ready in {time.perf_counter() - t0:.1f}s")
    STATUS["rag_loaded"] = True
    return {"gen": gen, "pipe": pipe}


@st.cache_resource(show_spinner="تحميل محلل الفجوات لأول مرة... (قد يُنزَّل المصنف من Hugging Face في أول مرة)")
def gap():
    out = {"gap_error": None, "analyzer": None, "retriever": None, "reqs": None}
    try:
        import gap_service as gs
        from analyzer_v2 import GapAnalyzerV2
        from checklist_v2 import load_requirements
        from rules_v2 import OPP_CATEGORY_MAP

        reqs = load_requirements(CHECKLIST)
        clf = None
        if os.getenv("USE_CLASSIFIER", "1") == "0":
            STATUS["clf_warning"] = "المصنف معطّل (USE_CLASSIFIER=0): التحليل بالقواعد فقط."
        else:
            t1 = time.perf_counter()
            try:
                Adapter, default_model = gs.import_classifier()
                source = gs.resolve_classifier_source(MODEL_DIR, os.getenv("CLASSIFIER_SOURCE"), default_model)
                if not source:
                    raise FileNotFoundError("no local model folder and no default Hugging Face repo")
                base = Adapter(source)
                STATUS["clf_source"] = str(source)
                if gs.apply_default_labels_if_generic(base):
                    STATUS["clf_notice"] = ("أسماء فئات المصنف عامة (LABEL_i): طُبّق ترتيب OPP-115 القياسي كافتراض. "
                                            "تأكدي منه مع من درّبت النموذج.")
                bad = gs.label_name_problems(base, OPP_CATEGORY_MAP)
                if bad:
                    STATUS["clf_warning"] = ("أسماء فئات المصنف لا تطابق خريطة OPP: سيُخفَّض الحكم إلى Partial بصمت. "
                                             "غير موجود: " + ", ".join(bad[:4]) + " ...")
                clf = gs.MemoClassifier(base)
                STATUS["clf_loaded"] = True
                _log(f"[load] XLM-R classifier ({source}) ready in {time.perf_counter() - t1:.1f}s")
            except Exception as exc:
                STATUS["clf_warning"] = (f"تعذّر تحميل المصنف ({type(exc).__name__}: {str(exc)[:120]}): "
                                         "التحليل بالقواعد فقط.")
                _log(f"[load] classifier unavailable: {type(exc).__name__}: {exc}")

        pipe = rag()["pipe"]                      # the retriever needs the RAG pipeline
        try:
            retriever = gs.make_cached_retriever(pipe, cache_path=RETRIEVAL_CACHE)
        except TypeError:                         # older gap_service.py without disk cache: update it!
            _log("[load] WARNING: gap_service.py is old (no cache_path). Replace it with the new version.")
            retriever = gs.make_cached_retriever(pipe)
        if os.getenv("SKIP_WARMUP") == "1":
            _log("[load] warm-up skipped (SKIP_WARMUP=1)")
        else:
            t2 = time.perf_counter()
            gs.warm_up(retriever, reqs, log=_log)
            _log(f"[load] checklist retrieval ready in {time.perf_counter() - t2:.1f}s")
        out.update(analyzer=GapAnalyzerV2(reqs, classifier=clf), retriever=retriever, reqs=reqs)
        STATUS["gap_loaded"] = True
    except Exception as exc:      # the Q&A tab must keep working if the gap files are missing or broken
        out["gap_error"] = f"{type(exc).__name__}: {exc}"
    return out


def preload():
    rag()
    gap()


def status():
    return dict(STATUS)