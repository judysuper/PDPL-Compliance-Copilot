"""SANAD | سند — Saudi PDPL Compliance Intelligence (Streamlit).

Tab 1  Ask the law (AR/EN): RAGPipeline.ask -> answer + article citations + retrieved text, or "not covered".
Tab 2  Privacy-policy gap analysis (team's GapAnalyzerV2): paste a policy -> Covered / Partial / Missing for the
       15 checklist requirements, with the evidence paragraph and the retrieved article text.

Run from the project root:   streamlit run app/streamlit_app.py
Live demo (load everything first):   $env:PRELOAD="1"; streamlit run app/streamlit_app.py     (PowerShell)
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
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gap_service as gs        # light module (standard library only)
import system_loader as SL      # lazy, cached loaders for the RAG pipeline and the gap analyzer

RAW_POLICIES = ROOT / "data" / "local_policies" / "raw"
GOLD_CSV = ROOT / "data" / "local_policies" / "gold" / "pdpl_local_policies_gold.csv"
ICON = {"Covered": "✅", "Partial": "🟡", "Missing": "❌"}

st.set_page_config(page_title="سند | SANAD AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {color-scheme:dark; --bg:#0B1022; --surface:#151D39; --surface2:#202849; --border:#354064; --text:#F5F4FF; --muted:#B7BFDB; --purple:#A88BFF; --red:#FF657F;}
html,body,[data-testid="stAppViewContainer"],.stApp {font-family:'Tajawal','Cairo','Segoe UI',Tahoma,sans-serif!important;}
.stApp {direction:rtl;text-align:right;background:radial-gradient(ellipse at 80% 0%,rgba(103,70,184,.26),transparent 40%),linear-gradient(160deg,#0B1022 0%,#111833 55%,#1C1740 100%)!important;color:var(--text)!important;}
[data-testid="stHeader"] {background:#0B0F1B!important;}
[data-testid="stToolbar"] button, [data-testid="stHeader"] button {color:#F5F4FF!important;}
.block-container {padding-top:1.35rem;max-width:1230px;padding-bottom:4rem;}
[data-testid="stSidebar"] {background:linear-gradient(175deg,#101A3C,#202354 65%,#332A69)!important;border-left:1px solid #504780;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#F5F2FF!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#C6C9E3!important;}
/* Default typography, including text in expanders, tabs, tables and controls */
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] h4,[data-testid="stAppViewContainer"] p,[data-testid="stAppViewContainer"] label,[data-testid="stAppViewContainer"] li {color:var(--text);}
[data-testid="stCaptionContainer"],small {color:var(--muted)!important;}
.sanad-hero {position:relative;overflow:hidden;border-radius:27px;padding:38px 42px;margin:5px 0 22px;color:#FFF!important;background:linear-gradient(120deg,#121C40 0%,#29265C 55%,#6545A8 100%);box-shadow:0 20px 45px rgba(0,0,0,.25);border:1px solid rgba(192,169,255,.22);}
.sanad-hero:before {content:"";position:absolute;width:310px;height:310px;border-radius:50%;top:-155px;left:-100px;background:radial-gradient(circle,rgba(202,160,255,.34),transparent 70%);}
.sanad-hero:after {content:"";position:absolute;width:190px;height:190px;border:1px solid rgba(255,255,255,.17);border-radius:32px;transform:rotate(27deg);left:65px;top:44px;}
.sanad-brand {display:flex;align-items:center;gap:14px;position:relative;z-index:1;}
.sanad-mark {display:grid;place-items:center;width:62px;height:62px;border:1px solid rgba(255,255,255,.3);border-radius:19px;background:rgba(255,255,255,.12);font-size:32px;}
.sanad-name {font-size:43px;font-weight:850;line-height:1.2;letter-spacing:-1px;color:#FFF;}
.sanad-en {color:#E2D5FF;font-size:13px;font-weight:700;letter-spacing:2px;direction:ltr;text-align:right;}
.sanad-hero p {color:#F3EFFF!important;font-size:16px;line-height:1.9;margin:18px 0 0;max-width:720px;position:relative;z-index:1;}
.sanad-chips {display:flex;gap:9px;flex-wrap:wrap;margin-top:23px;position:relative;z-index:1;}
.sanad-chip {border:1px solid rgba(255,255,255,.24);background:rgba(255,255,255,.09);border-radius:99px;padding:7px 14px;color:#F1EBFF;font-size:12px;}
.sanad-section {margin:10px 0 16px;}
.sanad-section h3 {font-size:21px;font-weight:800;margin:0 0 5px;color:#F5F4FF!important;}
.sanad-section p {color:#BEC6E4!important;margin:0;font-size:14px;}
.sanad-foot {margin-top:32px;border-top:1px solid #354064;padding-top:18px;text-align:center;color:#AEB8D9;font-size:12px;}
/* Tabs: text visible in inactive, active, hover and focus states */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #394166;padding-bottom:7px;}
button[data-baseweb="tab"],button[data-baseweb="tab"] * {color:#E3E5FF!important;opacity:1!important;}
button[data-baseweb="tab"] {border-radius:14px!important;background:#202849!important;border:1px solid #485077!important;padding:13px 24px!important;font-weight:750!important;}
button[data-baseweb="tab"][aria-selected="true"] {background:#513B89!important;border-color:#B39AFF!important;box-shadow:inset 0 -2px 0 #BFA8FF;}
button[data-baseweb="tab"][aria-selected="true"],button[data-baseweb="tab"][aria-selected="true"] * {color:#FFF!important;}
button[data-baseweb="tab"]:hover {background:#34305D!important;border-color:#AA90FA!important;}
button[data-baseweb="tab"]:focus-visible {outline:2px solid #C6B5FF!important;outline-offset:2px;}
/* All buttons, including download JSON and secondary actions */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button {background:#292E56!important;color:#FFF!important;border:1px solid #7971B8!important;border-radius:12px!important;font-weight:750!important;min-height:43px;}
.stButton>button *, .stDownloadButton>button * {color:#FFF!important;opacity:1!important;}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button {background:linear-gradient(110deg,#664AC7,#8C63E7)!important;border-color:#AD96FA!important;box-shadow:0 8px 18px rgba(103,71,200,.2);}
.stButton>button:hover,.stDownloadButton>button:hover {background:#4C3B83!important;border-color:#C3B1FF!important;color:#FFF!important;}
.stButton>button:disabled {background:#252C49!important;color:#B5BAD2!important;opacity:.75!important;}
.stDownloadButton>button {background:linear-gradient(110deg,#6245B8,#7959D4)!important;}
/* Input fields and dropdowns */
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-baseweb="select"]>div,[data-testid="stFileUploader"] {background:#171E37!important;color:#F7F5FF!important;border:1px solid #4A557A!important;border-radius:13px!important;}
textarea::placeholder,input::placeholder {color:#A6B1D0!important;opacity:1;}
[data-baseweb="select"] span,[data-baseweb="select"] div,[data-baseweb="select"] svg {color:#F7F5FF!important;fill:currentColor;}
[data-baseweb="popover"],[role="listbox"],[role="option"] {background:#202849!important;color:#F7F5FF!important;}
[role="option"]:hover {background:#42376C!important;}
[data-testid="stFileUploader"] section {background:#171E37!important;color:#F7F5FF!important;}
[data-testid="stFileUploader"] button {color:#FFF!important;background:#30395E!important;border-color:#6E75A3!important;}
/* Expander title and expanded content: never dark-on-dark */
[data-testid="stExpander"] {border:1px solid #3F496E!important;border-radius:14px!important;background:#171F3A!important;margin-bottom:8px;overflow:hidden;}
[data-testid="stExpander"] details,[data-testid="stExpander"] summary,[data-testid="stExpander"] [data-testid="stExpanderDetails"] {background:#171F3A!important;color:#F5F4FF!important;}
[data-testid="stExpander"] summary:hover {background:#283052!important;}
[data-testid="stExpander"] summary *,[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span,[data-testid="stExpander"] summary svg {color:#F7F5FF!important;fill:currentColor;opacity:1!important;}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,[data-testid="stExpander"] [data-testid="stExpanderDetails"] span {color:#E9EBFF!important;}
[data-testid="stMetric"] {background:#1C2543!important;border:1px solid #404A73;padding:16px;border-radius:17px;}
[data-testid="stMetricValue"],[data-testid="stMetricLabel"] {color:#F5F4FF!important;}
/* Streamlit notifications: distinct but readable warning, info, success, error */
[data-testid="stAlert"] {border-radius:13px!important;background:#242C4C!important;border:1px solid #525D83!important;}
[data-testid="stAlert"] *,[data-testid="stAlert"] p {color:#F7F4FF!important;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color:#F7F4FF!important;}
[data-testid="stDataFrame"] {border-radius:13px;overflow:hidden;border:1px solid #465077;}
.sanad-caution {background:#401D35;color:#FFD9E1;border:1px solid #9D3B60;border-right:4px solid #FF657F;border-radius:13px;padding:13px 17px;font-size:13px;margin-bottom:24px;}
.sanad-side-brand {font-size:27px;font-weight:850;margin:10px 0 4px;color:#FFF;}
.sanad-side-sub {font-size:12px;opacity:.85;margin-bottom:25px;color:#D2D1F3;}
@media(max-width:650px) {.sanad-hero{padding:27px 22px;border-radius:19px;}.sanad-name{font-size:34px;}.sanad-hero p{font-size:14px;}.block-container{padding-top:.8rem;}button[data-baseweb="tab"]{padding:10px 12px!important;}}
</style>
""", unsafe_allow_html=True)

# Demo day: PRELOAD=1 loads the RAG pipeline and the gap analyzer before the first click.
if os.getenv("PRELOAD") == "1":
    SL.preload()

with st.sidebar:
    st.markdown('<div class="sanad-side-brand">◈ سند <span style="font-size:12px;letter-spacing:2px">SANAD AI</span></div><div class="sanad-side-sub">منصة ذكاء الامتثال لنظام حماية البيانات الشخصية</div>', unsafe_allow_html=True)
    st.header("⚙️ الإعدادات")
    options = ["gemini", "ollama", "mock"]
    default = os.getenv("LLM_PROVIDER", "gemini")
    provider = st.selectbox("مزوّد النموذج اللغوي (للأسئلة)", options, index=options.index(default) if default in options else 0)
    _st = SL.status()
    st.caption("محرك الأسئلة: " + ("جاهز" if _st["rag_loaded"] else "يُحمَّل عند أول سؤال"))
    if _st["gap_loaded"]:
        st.caption("محلل الفجوات: جاهز | " + ("القواعد + مصنف XLM-R" if _st["clf_loaded"] else "قواعد فقط (المصنف غير محمّل)"))
    else:
        st.caption("محلل الفجوات: يُحمَّل عند أول تحليل")
    if _st["clf_loaded"] and _st["clf_source"]:
        st.caption("مصدر المصنف: " + _st["clf_source"])
    if _st["clf_notice"]:
        st.info(_st["clf_notice"])
    if _st["clf_warning"]:
        st.warning(_st["clf_warning"])

st.markdown("""
<section class="sanad-hero">
  <div class="sanad-brand"><div class="sanad-mark">◈</div><div><div class="sanad-name">سند</div><div class="sanad-en">SANAD AI</div></div></div>
  <p>امتثالك مستند إلى الدليل. مساعد ذكي لفهم نظام حماية البيانات الشخصية السعودي، واستكشاف الفجوات في سياسات الخصوصية، وربط النتائج بالمواد النظامية.</p>
  <div class="sanad-chips"><span class="sanad-chip">✦ ذكاء قانوني مدعوم بالمصادر</span><span class="sanad-chip">◈ تحليل فجوات الخصوصية</span><span class="sanad-chip">▤ استشهادات بالأدلة</span></div>
</section>
<div class="sanad-caution">⚠️ <strong>تنبيه قانوني:</strong> سند أداة تحليل ومساعدة أولية، وليست استشارة قانونية. تصنيف Covered لا يعني ثبوت الامتثال الفعلي.</div>
""", unsafe_allow_html=True)
tab_qa, tab_gap = st.tabs(["✦ اسأل سند", "▤ تحليل فجوات الخصوصية"])

# ----------------------------------------------------------------------------- tab 1: Q&A
with tab_qa:
    st.markdown('<div class="sanad-section"><h3>اسأل سند عن النظام</h3><p>اكتب سؤالك بالعربية أو الإنجليزية، واحصل على إجابة مدعومة بالنصوص النظامية المسترجعة.</p></div>', unsafe_allow_html=True)
    q = st.text_area("اكتب سؤالك (عربي أو English)", height=90, placeholder="هل يجوز نقل بيانات العملاء إلى خارج المملكة؟")
    if st.button("✦ احصل على إجابة مستندة", type="primary", disabled=not q.strip()):
        try:
            with st.spinner("استرجاع المواد وتوليد الإجابة..."):
                res = SL.rag()["pipe"].ask(q.strip(), provider=provider)
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
    st.markdown('<div class="sanad-section"><h3>محلل فجوات الخصوصية</h3><p>ارفع سياسة خصوصية أو الصق نصها، وسيقارن سند محتواها بمتطلبات قائمة التحقق.</p></div>', unsafe_allow_html=True)
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

    if st.button("◈ ابدأ تحليل السياسة", type="primary", disabled=not text.strip()):
        G = SL.gap()                                    # loads the analyzer the first time only
        if G["gap_error"]:
            st.error("وحدة تحليل الفجوات لم تُحمَّل: " + G["gap_error"])
            st.stop()
        with st.spinner("تحليل السياسة..."):
            rep = gs.run_gap(G["analyzer"], text, G["retriever"])
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


st.markdown('<div class="sanad-foot">سند | SANAD AI · Saudi PDPL Compliance Intelligence · نتائج استرشادية تستلزم مراجعة بشرية</div>', unsafe_allow_html=True)