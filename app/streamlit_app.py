"""PDPL Compliance Copilot - live demo (Streamlit).

Tab 1  Ask the law (AR/EN): RAGPipeline.ask -> answer + article citations + retrieved text, or "not covered".
Tab 2  Privacy-policy gap analysis (team's GapAnalyzerV2): paste a policy -> Covered / Partial / Missing for the
       15 checklist requirements, with the evidence paragraph and the retrieved article text.

Run from the project root:   streamlit run app/streamlit_app.py
Env: LLM_PROVIDER (gemini | ollama | mock). Never commit API keys (use .env).
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "gap_analyzer", ROOT / "app"):   # gap_analyzer modules use flat imports
    sys.path.insert(0, str(p))

CHECKLIST = ROOT / "data" / "pdpl_checklist" / "member1_pdpl_compliance_checklist.csv"
MODEL_DIR = ROOT / "models" / "xlm_roberta_best"
RAW_POLICIES = ROOT / "data" / "local_policies" / "raw"
GOLD_CSV = ROOT / "data" / "local_policies" / "gold" / "pdpl_local_policies_gold.csv"
ICON = {"Covered": "✅", "Partial": "🟡", "Missing": "❌"}

st.set_page_config(page_title="PDPL Compliance Copilot", page_icon="⚖️", layout="wide")
st.markdown("<style>.stApp{direction:rtl;text-align:right} textarea{direction:auto}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="تحميل الفهارس والنماذج... (مرة واحدة)")
def load_system():
    """Heavy objects are built ONCE per server. Article retrieval for the checklist does not depend on the
    policy, so it is warmed up here and costs 0 seconds during the demo."""
    from generation import generation as gen
    from rag_pipeline import RAGPipeline

    sysd = {"gen": gen, "pipe": RAGPipeline(), "gap_error": None, "analyzer": None, "reqs": None,
            "retriever": None, "clf_warning": None, "clf_loaded": False}
    try:
        import gap_service as gs
        from analyzer_v2 import GapAnalyzerV2
        from checklist_v2 import load_requirements
        from rules_v2 import OPP_CATEGORY_MAP

        reqs = load_requirements(CHECKLIST)
        clf = None
        if MODEL_DIR.exists():
            try:
                from classifier_v2 import XLMRClassifierAdapter
                base = XLMRClassifierAdapter(MODEL_DIR)
                bad = gs.label_name_problems(base, OPP_CATEGORY_MAP)
                if bad:
                    sysd["clf_warning"] = ("أسماء فئات المصنف لا تطابق خريطة OPP (مثل LABEL_0). "
                                           "سيُخفَّض الحكم إلى Partial بصمت. غير موجود: " + ", ".join(bad[:4]) + " ...")
                clf = gs.MemoClassifier(base)
                sysd["clf_loaded"] = True
            except Exception as exc:
                sysd["clf_warning"] = f"تعذّر تحميل المصنف ({type(exc).__name__}): التحليل بالقواعد فقط."
        retriever = gs.make_cached_retriever(sysd["pipe"])
        gs.warm_up(retriever, reqs)
        sysd.update(analyzer=GapAnalyzerV2(reqs, classifier=clf), reqs=reqs, retriever=retriever, gs=gs)
    except Exception as exc:    # the Q&A tab must still work if the gap files are missing or broken
        sysd["gap_error"] = f"{type(exc).__name__}: {exc}"
    return sysd


S = load_system()

with st.sidebar:
    st.header("الإعدادات")
    options = ["gemini", "ollama", "mock"]
    default = os.getenv("LLM_PROVIDER", "gemini")
    provider = st.selectbox("مزوّد النموذج اللغوي (للأسئلة)", options, index=options.index(default) if default in options else 0)
    st.caption(f"مصنف XLM-R: {'محمّل' if S['clf_loaded'] else 'غير محمّل (قواعد فقط)'}")
    if S["clf_warning"]:
        st.warning(S["clf_warning"])

st.title("⚖️ مساعد الامتثال لنظام حماية البيانات الشخصية (PDPL)")
st.warning("⚠️ هذه أداة مساعدة أولية وليست استشارة قانونية. 'Covered' لا يعني الامتثال الفعلي.")
tab_qa, tab_gap = st.tabs(["❓ اسأل النظام", "📋 تحليل سياسة الخصوصية"])

# ----------------------------------------------------------------------------- tab 1: Q&A
with tab_qa:
    q = st.text_area("اكتب سؤالك (عربي أو English)", height=90, placeholder="هل يجوز نقل بيانات العملاء إلى خارج المملكة؟")
    if st.button("اسأل", type="primary", disabled=not q.strip()):
        try:
            with st.spinner("استرجاع المواد وتوليد الإجابة..."):
                res = S["pipe"].ask(q.strip(), provider=provider)
        except Exception as exc:
            st.error(f"تعذّر توليد الإجابة ({type(exc).__name__}). جرّب مزوّد mock أو ollama من الشريط الجانبي.")
            st.stop()
        if res["abstained"]:
            st.warning(res["answer"])
            st.caption(f"سبب الامتناع: {res.get('abstention_reason')} | أعلى درجة: {res.get('max_rerank_score')} "
                       f"| العتبة: {res.get('threshold')}")
        else:
            st.success(res["answer"])
            chk = res.get("citation_check") or {}
            st.caption(f"فحص الاستشهادات: صالح={chk.get('valid')} | المذكورة={chk.get('cited_articles')} "
                       f"| غير المدعومة={chk.get('unsupported_articles')}")
        st.subheader("النصوص المسترجعة (الدليل)")
        for i, d in enumerate(res.get("final_sources") or res.get("reranked") or [], 1):
            score = d.get("rerank_score")
            title = f"[{i}] المادة {d.get('article_number')} | {d.get('source')}"
            if isinstance(score, (int, float)):
                title += f" | درجة: {score:.2f}"
            with st.expander(title, expanded=(i == 1)):
                st.write(d.get("article_text") or "(النص غير مرفق في هذه النتيجة)")
        st.caption("الأزمنة (ث): " + ", ".join(f"{k}={v:.2f}" for k, v in res["timings"].items()))
        st.caption(res["disclaimer"])

# ----------------------------------------------------------------------------- tab 2: gap analysis
with tab_gap:
    if S["gap_error"]:
        st.info("وحدة تحليل الفجوات لم تُحمَّل. تأكدي من وجود src/gap_analyzer/*.py و"
                "data/pdpl_checklist/member1_pdpl_compliance_checklist.csv.")
        st.code(S["gap_error"])
        st.stop()
    gs = S["gs"]
    gold = gs.load_gold(GOLD_CSV)
    samples = sorted(RAW_POLICIES.glob("*.txt")) if RAW_POLICIES.exists() else []
    col1, col2 = st.columns([2, 1])
    with col2:
        pick = st.selectbox("سياسة جاهزة (للتدرّب على العرض)", ["—"] + [p.stem for p in samples])
        up = st.file_uploader("أو ارفعي ملف .txt", type=["txt"])
    with col1:
        default_text = (RAW_POLICIES / f"{pick}.txt").read_text(encoding="utf-8", errors="replace") if pick != "—" else ""
        if up is not None:
            default_text = up.read().decode("utf-8", errors="ignore")
        text = st.text_area("الصق نص سياسة الخصوصية هنا", value=default_text, height=260)

    if st.button("حلّل السياسة", type="primary", disabled=not text.strip()):
        with st.spinner("تحليل السياسة..."):
            rep = gs.run_gap(S["analyzer"], text, S["retriever"])
        for w in rep["warnings"]:
            st.warning(w)
        c = st.columns(4)
        c[0].metric("✅ Covered", rep["summary"]["Covered"])
        c[1].metric("🟡 Partial", rep["summary"]["Partial"])
        c[2].metric("❌ Missing", rep["summary"]["Missing"])
        c[3].metric("الزمن (ث)", rep["seconds"])

        name = gs.policy_name_from_stem(pick) if pick != "—" and not up and text == default_text else None
        rows = []
        for r in rep["results"]:
            g = gold.get((name, r["requirement_id"])) if name else None
            rows.append({"البند": r["requirement_id"], "الحالة": f"{ICON[r['status']]} {r['status']}",
                         "الكلمات المطابقة": r["matched_terms"], "فئة المصنف": r["classifier_category"],
                         "درجة المصنف": r["classifier_score"], "المادة": f"{r['source_document']} / {r['article_number']}",
                         **({"المرجع (Gold)": g, "اتفاق": "✔" if g == r["status"] else "✘"} if g else {})})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if name and any(g for g in (gold.get((name, r["requirement_id"])) for r in rep["results"])):
            agree = sum(gold.get((name, r["requirement_id"])) == r["status"] for r in rep["results"])
            st.caption(f"اتفاق مع الحكم المرجعي لهذه السياسة: {agree}/{len(rep['results'])} "
                       "(للتدرّب فقط: هذه السياسة من مجموعة التطوير).")

        for r in rep["results"]:
            with st.expander(f"{ICON[r['status']]} {r['requirement_id']} — {r['requirement']}"):
                st.write(f"**المرجع:** {r['source_document']} — المادة {r['article_number']} | "
                         f"**سؤال التحقق:** {r['verification_question']}")
                if r["evidence_text"]:
                    st.info(f"فقرة الدليل: {r['evidence_text']}")
                else:
                    st.write("لا توجد فقرة دليل.")
                if r["retrieved_article_text"]:
                    st.caption("نص المادة المسترجع:")
                    st.write(r["retrieved_article_text"][:900])
                elif r["article_retrieved"] is False:
                    st.caption("⚠ لم تُسترجع مادة الـ Checklist لهذا البند. أعلى النتائج: " + "؛ ".join(r["top_retrieved"]))
                if r["requirement_id"] in ("REQ-012", "REQ-014"):
                    st.caption("ملاحظة: دليل هذا البند وثيقة داخلية، ولا يُحكم عليه 'Covered' إلا إذا ذُكر صراحة في السياسة.")
        st.download_button("تنزيل التقرير JSON", json.dumps(rep, ensure_ascii=False, indent=2),
                           file_name="gap_report.json", mime="application/json")
        st.caption("هذا تقرير أولي آلي وليس استشارة قانونية.")
