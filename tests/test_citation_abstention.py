"""Day-3 tests: citation checker and abstention (no API key, no models needed).

Article texts below are short excerpts of the official texts, copied from the
`supporting_text` column of data/evaluation/gold_qa_set.csv, and the questions are the
team's gold questions (question ids in comments). Model answers are written the
way the LLM answers with the day-2 prompt; wrong answers reproduce typical LLM
mistakes (citing a real article that was not retrieved, mixing in GDPR, pairing a
tag with the wrong article).

The only assumed values are the reranker scores in test 5 and in the threshold
tests; real scores come from the calibration section of the day-3 notebook.

The first six tests are the six cases of the day-3 plan, in the same order.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation.abstention import (  # noqa: E402
    build_answer_prompt,
    calibrate_threshold,
    evidence_gate,
    guarded_answer,
    is_not_covered,
)
from generation.citation_checker import (  # noqa: E402
    check_citations,
    extract_articles,
    normalize_article_number,
)

GOLD_PATH = next((p for p in (ROOT / "data" / "evaluation" / "gold_qa_set.csv",
                              ROOT / "data" / "qa" / "gold_qa_set.csv") if p.exists()),
                 ROOT / "data" / "evaluation" / "gold_qa_set.csv")

# ------------------------------------------------- official article excerpts
PDPL_4_AR = ("يكون لصاحب البيانات الشخصية -وفقاً للأحكام الواردة في النظام وما تحدده اللوائح- "
             "الحقوق الآتية: 1- الحق في العلم، ويشمل ذلك إحاطته علماً بالمسوغ النظامي لجمع "
             "بياناته الشخصية والغرض من جمعها. 2- الحق في وصوله إلى بياناته الشخصية المتوفرة "
             "لدى جهة التحكم، وفق الضوابط والإجراءات التي تحددها اللوائح، ودون إخلال بما ورد "
             "في المادة (التاسعة) من النظام.")
PDPL_5_AR = ("1- فيما عدا الأحوال المنصوص عليها في النظام، لا تجوز معالجة البيانات الشخصية أو "
             "تغيير الغرض من معالجتها إلاّ بعد موافقة صاحبها. 2- في جميع الأحوال، يجوز لصاحب "
             "البيانات الشخصية الرجوع عن الموافقة المشار إليها في الفقرة (1) من هذه المادة في "
             "أي وقت، وتحدد اللوائح الضوابط اللازمة لذلك.")
PDPL_6_AR = ("لا تخضع معالجة البيانات الشخصية للموافقة المشار إليها في الفقرة (1) من المادة "
             "(الخامسة) من النظام، في الأحوال الآتية: 1- عندما تُحقق المعالجة مصلحة متحققة "
             "لصاحب البيانات وكان الاتصال به متعذراً أو كان من الصعب تحقيق ذلك. 2- عندما تكون "
             "المعالجة بمقتضى نظام آخر أو تنفيذاً لاتفاق سابق يكون صاحب البيانات الشخصية طرفاً فيه.")
PDPL_7_EN = ("The consent referred to in paragraph (1) of Article (5) of this Law may not form a "
             "condition of providing a service or a benefit, unless such service or benefit is "
             "directly related to the Processing of Personal Data for which the consent is given.")
IMPL_24_AR = ("1- تُشعر جهة التحكم الجهة المختصة في حالة وقوع حادثة تسرّب للبيانات الشخصية خلال مدة "
              "لا تتجاوز (72) ساعة من وقت علمها بالحادثة، إذا كان من شأن تلك الحادثة الإضرار "
              "بالبيانات الشخصية أو صاحب البيانات الشخصية أو كانت تتعارض مع حقوقه أو مصالحه.")
TRANSFER_7_EN = ("１. The controller shall conduct a risk assessment before transferring or disclosing "
                 "personal data to a party outside the Kingdom in the following cases: A. Transfer or "
                 "disclosure of personal data to a party outside the Kingdom in accordance with "
                 "Article (4) of the Regulation. B. Transferring or disclosing sensitive data to "
                 "parties outside the Kingdom on a continuous basis or on a large scale.")


def article(number, text, source="pdpl", lang="ar", title="", score=None):
    d = {"article_number": str(number), "article_text": text, "article_title": title,
         "language": lang, "source": source}
    if score is not None:
        d["rerank_score"] = score
    return d


def retrieved_for_q005(with_scores=False):
    """What retrieval typically returns for Q005 (withdrawing consent): articles 5 and 6."""
    s5, s6 = (0.91, 0.47) if with_scores else (None, None)
    return [article(5, PDPL_5_AR, score=s5), article(6, PDPL_6_AR, score=s6)]


def llm_must_not_be_called(prompt):
    raise AssertionError("the LLM should not have been called")


def scripted_llm(*replies):
    """Fake LLM returning the given replies in order and recording the prompts."""
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return replies[min(len(calls), len(replies)) - 1]

    fn.calls = calls
    return fn


Q005 = "وافقت مستخدمة تطبيق على معالجة بياناتها ثم رغبت في سحب الموافقة. هل يتيح لها النظام الرجوع عنها؟"
Q005_GOOD = ("نعم، يجوز لصاحب البيانات الشخصية الرجوع عن الموافقة في أي وقت [Source 1] (Article 5)، "
             "وتحدد اللوائح الضوابط اللازمة لذلك [Source 1] (Article 5). ولا يشمل ذلك الأحوال التي "
             "لا تخضع فيها المعالجة للموافقة أصلًا [Source 2] (Article 6).")


# ============================================== the six cases of the plan
def test_1_valid_citation_is_accepted():
    report = check_citations(Q005_GOOD, retrieved_for_q005())
    assert report["valid"]
    assert report["supported_articles"] == ["5", "6"]
    assert report["unsupported_articles"] == []


def test_2_citation_to_unretrieved_article_is_unsupported():
    # The model adds Article 18 (a real PDPL article on retention) that was NOT retrieved.
    answer = Q005_GOOD + " وبعد الرجوع يجب إتلاف البيانات وفق المادة 18 من النظام."
    report = check_citations(answer, retrieved_for_q005())
    assert not report["valid"]
    assert report["cited_articles"] == ["5", "6", "18"]
    assert report["supported_articles"] == ["5", "6"]
    assert report["unsupported_articles"] == ["18"]


def test_3_answer_without_required_citation_is_detected():
    no_tags = "نعم، يجوز لصاحب البيانات الرجوع عن موافقته في أي وقت وفق الضوابط التي تحددها اللوائح."
    report = check_citations(no_tags, retrieved_for_q005())
    assert not report["valid"] and not report["has_citations"]
    # citing "المادة الخامسة" without a [Source n] tag also breaks the required format
    assert not check_citations("نعم، وفق المادة الخامسة يجوز الرجوع.", retrieved_for_q005())["valid"]


def test_4_abstains_when_nothing_was_retrieved():
    # Q040: no article can answer an audit-statistics question
    out = guarded_answer("How many Saudi companies were audited for PDPL compliance in 2025?",
                         [], llm_fn=llm_must_not_be_called)
    assert out["abstained"] and out["abstention_reason"] == "no_documents"
    assert "I cannot provide a reliable answer" in out["answer"]


def test_5_abstains_when_scores_are_below_the_threshold():
    # Q032 asks for 2025 breach statistics; the nearest articles are about breach notification
    docs = [article(24, IMPL_24_AR, source="implementing", score=0.21),
            article(20, "تُشعر جهة التحكم الجهة المختصة ...", score=0.08)]
    out = guarded_answer("كم عدد البلاغات عن تسرب البيانات الشخصية التي تلقتها الجهة المختصة "
                         "في المملكة خلال عام 2025؟", docs,
                         min_rerank_score=0.35, llm_fn=llm_must_not_be_called)
    assert out["abstained"] and out["abstention_reason"] == "low_retrieval_score"
    assert out["max_score"] == pytest.approx(0.21)


def test_6_supported_question_is_answered_with_verifiable_citations():
    llm = scripted_llm(Q005_GOOD)
    out = guarded_answer(Q005, retrieved_for_q005(with_scores=True), min_rerank_score=0.35, llm_fn=llm)
    assert not out["abstained"]
    assert out["citation_check"]["valid"]
    assert [d["article_number"] for d in out["sources_used"]] == ["5", "6"]
    assert len(llm.calls) == 1


# ===================================== realistic checker cases
def test_gdpr_article_mixed_into_the_answer_is_caught():
    # typical hallucination: the model fills a gap with GDPR's "right to be forgotten"
    answer = Q005_GOOD + " ويقابل ذلك الحق في النسيان في المادة 17 من GDPR."
    report = check_citations(answer, retrieved_for_q005())
    assert report["unsupported_articles"] == ["17"] and not report["valid"]


def test_wrong_tag_for_a_retrieved_article_is_caught():
    # Article 6 was retrieved, but as Source 2: "[Source 1] (Article 6)" is a wrong pairing
    report = check_citations("لا تحتاج كل معالجة إلى موافقة [Source 1] (Article 6).", retrieved_for_q005())
    assert report["unsupported_articles"] == []
    assert report["mismatched_pairs"] == [{"source": 1, "cited_article": "6", "actual_article": "5"}]
    assert not report["valid"]


def test_cross_reference_inside_retrieved_text_is_a_warning_not_an_error():
    # Q007: only Article 7 is retrieved; its text refers to Article (5). Repeating that is not invented.
    docs = [article(7, PDPL_7_EN, lang="en")]
    answer = ("Consent referred to in Article 5 may not be a condition for providing a service "
              "unless the service is directly related to the processing [Source 1] (Article 7).")
    report = check_citations(answer, docs)
    assert report["valid"]
    assert report["cross_referenced_articles"] == ["5"] and report["warnings"]
    assert not check_citations(answer, docs, allow_cross_references=False)["valid"]


def test_same_number_in_two_documents_follows_the_source_tag():
    # PDPL Article 24 and Implementing Regulation Article 24 are different articles
    docs = [article(24, IMPL_24_AR, source="implementing"), article(5, PDPL_5_AR)]
    report = check_citations("يجب الإشعار خلال 72 ساعة [Source 1] (Article 24).", docs)
    assert report["valid"] and report["cited_sources"] == [1]


def test_english_answer_with_full_width_digits_in_the_source():
    docs = [article(7, TRANSFER_7_EN, source="transfer", lang="en")]
    answer = ("A risk assessment is required before a transfer when sensitive data is transferred "
              "continuously or on a large scale [Source 1] (Article 7).")
    report = check_citations(answer, docs)
    assert report["valid"]
    assert report["claims"][0]["overlap"] > 0.5


def test_arabic_citation_styles_seen_in_model_answers():
    assert extract_articles("وفق المادة (٥) من النظام") == ["5"]
    assert extract_articles("نصت المادة السادسة على") == ["6"]
    assert extract_articles("المادة الحادية عشرة، الفقرة الثالثة") == ["11"]
    assert extract_articles("المادة الرابعة والعشرون من اللائحة") == ["24"]
    assert extract_articles("المواد 5 و6") == ["5", "6"]
    assert extract_articles("Articles 5, 6 and 7") == ["5", "6", "7"]
    assert check_citations("[المصدر 1] (المادة 5)", retrieved_for_q005())["valid"]


def test_float_article_numbers_from_csv_are_normalised():
    assert check_citations("[Source 1] (Article 5)", [article("5.0", PDPL_5_AR)])["valid"]


def test_overlap_is_none_across_languages():
    report = check_citations("يجب تقويم المخاطر قبل النقل [Source 1] (Article 7).",
                             [article(7, TRANSFER_7_EN, source="transfer", lang="en")])
    assert report["claims"][0]["overlap"] is None


# =================================== real data: the team's gold set (if present)
@pytest.mark.skipif(not GOLD_PATH.exists(), reason="data/evaluation/gold_qa_set.csv not found")
def test_extractor_reads_every_article_heading_in_the_gold_set():
    import pandas as pd
    gold = pd.read_csv(GOLD_PATH, encoding="utf-8-sig")
    gold = gold[gold["answerable_from_sources"].astype(str).str.lower() == "true"]
    failures = []
    for row in gold.itertuples():
        heading = str(row.supporting_text).split("\n")[0][:80]   # "المادة الحادية عشرة:" / "Article 7"
        if normalize_article_number(row.article_number) not in extract_articles(heading):
            failures.append((row.question_id, heading))
    assert not failures, failures


@pytest.mark.skipif(not GOLD_PATH.exists(), reason="data/evaluation/gold_qa_set.csv not found")
def test_reference_answers_cite_only_their_own_or_cross_referenced_articles():
    """Each article number in a reference answer is the gold article or is referenced in its text."""
    import pandas as pd
    gold = pd.read_csv(GOLD_PATH, encoding="utf-8-sig")
    gold = gold[gold["answerable_from_sources"].astype(str).str.lower() == "true"]
    failures = []
    for row in gold.itertuples():
        docs = [{"article_number": row.article_number, "article_text": str(row.supporting_text)}]
        report = check_citations(row.reference_answer, docs, require_source_tags=False)
        if report["unsupported_articles"]:
            failures.append((row.question_id, report["unsupported_articles"]))
    assert not failures, failures


# ================================================ abstention cases
def test_not_covered_detected_even_with_markdown():
    assert is_not_covered("NOT_COVERED: لا تتضمن المصادر قيمة الغرامة.")
    assert is_not_covered("**NOT_COVERED:** المصادر لا تغطي هذه الواقعة")
    assert not is_not_covered(Q005_GOOD)


def test_llm_not_covered_reply_leads_to_abstention():
    # Q034: a fine imposed on a specific company is a fact, not something the law text contains
    llm = scripted_llm("NOT_COVERED: لا تتضمن المصادر المسترجعة قرارات عقوبات فعلية أو قيمها.")
    docs = [article(35, "يعاقب بغرامة لا تزيد على ...", score=0.62)]
    out = guarded_answer("ما قيمة الغرامة التي فُرضت بالفعل على شركة معينة في الرياض؟", docs, llm_fn=llm)
    assert out["abstained"] and out["abstention_reason"] == "llm_not_covered"
    assert "قرارات عقوبات فعلية" in out["answer"]  # the LLM's explanation is kept


def test_unretrieved_article_is_repaired_by_one_retry():
    bad = Q005_GOOD + " ويجب الإتلاف بعد الرجوع وفق المادة 18."
    llm = scripted_llm(bad, Q005_GOOD)
    out = guarded_answer(Q005, retrieved_for_q005(), llm_fn=llm)
    assert not out["abstained"]
    assert out["first_attempt_valid"] is False and out["n_attempts"] == 2
    assert "[Source 1] (Article 5)" in llm.calls[1] and "rejected" in llm.calls[1]


def test_abstains_when_retry_still_cites_unsupported_articles():
    bad = Q005_GOOD + " ويقابل ذلك المادة 17 من GDPR."
    llm = scripted_llm(bad)
    out = guarded_answer(Q005, retrieved_for_q005(), llm_fn=llm, max_retries=1)
    assert out["abstained"] and out["abstention_reason"] == "unsupported_citations"
    assert len(llm.calls) == 2


def test_abstains_when_answer_has_no_citations():
    out = guarded_answer(Q005, retrieved_for_q005(),
                         llm_fn=scripted_llm("نعم، يمكنها الرجوع عن الموافقة في أي وقت."), max_retries=0)
    assert out["abstained"] and out["abstention_reason"] == "no_citations"


def test_threshold_drops_weak_articles_but_keeps_strong_ones():
    kept, reason, best = evidence_gate(retrieved_for_q005(with_scores=True), min_score=0.5)
    assert reason is None and [d["article_number"] for d in kept] == ["5"]


def test_threshold_without_reranker_scores_raises():
    with pytest.raises(ValueError):
        evidence_gate(retrieved_for_q005(), min_score=0.3)


def test_prompt_without_abstain_instruction_for_e5():
    from generation.generation import prompt_text
    for style in ("single", "system", "system_examples", "system_v2"):
        with_rule = prompt_text(build_answer_prompt(Q005, retrieved_for_q005(), True, style))
        without = prompt_text(build_answer_prompt(Q005, retrieved_for_q005(), False, style))
        assert "NOT_COVERED" in with_rule, style
        assert "NOT_COVERED" not in without, style
        assert "[Source 2]" in with_rule and "Article: 6" in with_rule


# ============================================ prompt styles (phase 1)
def test_prompt_styles_put_the_rules_where_expected():
    from generation.generation import ABSTAIN_REMINDER, RULES, build_messages
    docs = retrieved_for_q005()

    single = build_messages(Q005, docs, style="single")
    assert single["system"] is None and single["user"].startswith(RULES)

    system = build_messages(Q005, docs, style="system")
    assert system["system"] == RULES
    assert system["user"].endswith(ABSTAIN_REMINDER) and Q005 in system["user"]

    examples = build_messages(Q005, docs, style="system_examples")
    assert examples["system"].startswith(RULES) and "A: NOT_COVERED:" in examples["system"]


def test_unknown_prompt_style_is_rejected():
    from generation.generation import build_messages
    with pytest.raises(ValueError):
        build_messages(Q005, retrieved_for_q005(), style="fancy")


def test_guarded_answer_sends_the_system_message_to_the_provider(monkeypatch):
    sent = {}

    def fake_call_llm(prompt, provider=None, model=None, system=None):
        sent.update(prompt=prompt, provider=provider, system=system)
        return Q005_GOOD

    monkeypatch.setattr("generation.abstention.call_llm", fake_call_llm)
    out = guarded_answer(Q005, retrieved_for_q005(), provider="ollama", prompt_style="system")
    assert not out["abstained"] and out["prompt_style"] == "system"
    assert sent["provider"] == "ollama" and sent["system"].startswith("You are a Saudi")
    assert "NOT_COVERED" not in sent["prompt"].split("User question")[0]  # rules are not repeated


def test_repair_prompt_keeps_the_system_message():
    from generation.abstention import repair_prompt
    base = build_answer_prompt(Q005, retrieved_for_q005(), True, "system")
    report = {"issues": ["articles not in the retrieved sources: ['18']"]}
    fixed = repair_prompt(base, report, retrieved_for_q005())
    assert fixed["system"] == base["system"]
    assert "[Source 2] (Article 6)" in fixed["user"] and "NOT_COVERED" in fixed["user"]
    plain = repair_prompt(base, report, retrieved_for_q005(), abstain_instruction=False)
    assert "NOT_COVERED" not in plain["user"].split("IMPORTANT")[1]


# ============================================ Ollama / OpenAI providers
def _fake_post(captured, prompt_eval_count=600):
    import types

    def post(url, json, timeout):
        captured.update(url=url, json=json)
        return types.SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"response": Q005_GOOD, "prompt_eval_count": prompt_eval_count})
    return post


def test_ollama_request_sets_context_system_and_temperature(monkeypatch):
    from generation.generation import LAST_LLM, call_llm
    captured = {}
    monkeypatch.setattr("requests.post", _fake_post(captured))
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)

    assert call_llm("USER", provider="ollama", system="SYSTEM") == Q005_GOOD
    body = captured["json"]
    assert body["model"] == "command-r7b-arabic"
    assert body["system"] == "SYSTEM" and body["prompt"] == "USER"
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}  # not Ollama's 4096 default
    assert body["keep_alive"] == "0"  # unload after each answer (stable on a Colab T4)
    assert LAST_LLM["model"] == "command-r7b-arabic" and LAST_LLM["prompt_tokens"] == 600


def test_ollama_warns_when_the_prompt_fills_the_context(monkeypatch):
    import io
    import contextlib
    from generation.generation import call_llm
    captured = {}
    monkeypatch.setattr("requests.post", _fake_post(captured, prompt_eval_count=4096))
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        call_llm("USER", provider="ollama")
    assert "may have been cut" in out.getvalue()
    assert "system" not in captured["json"]  # no empty system field for style "single"


def test_openai_provider_sends_system_message(monkeypatch):
    import types
    from generation.generation import call_llm
    sent = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            msg = types.SimpleNamespace(content=Q005_GOOD)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeOpenAI:
        chat = types.SimpleNamespace(completions=FakeCompletions())

    openai = types.ModuleType("openai")
    openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert call_llm("USER", provider="openai", system="SYSTEM") == Q005_GOOD
    assert sent["model"] == "gpt-4o-mini" and sent["temperature"] == 0
    assert sent["messages"] == [{"role": "system", "content": "SYSTEM"},
                                {"role": "user", "content": "USER"}]


# ============================================ choosing the prompt style (phase 1b)
def test_choose_prompt_style_follows_the_pre_registered_rule():
    import pandas as pd
    from evaluation.eval_abstention import choose_prompt_style

    table = pd.DataFrame([
        {"prompt_style": "single", "abstention_accuracy": 0.2, "false_refusal_rate": 0.0,
         "first_attempt_citation_valid_rate": 0.9},
        {"prompt_style": "system", "abstention_accuracy": 0.8, "false_refusal_rate": 0.04,
         "first_attempt_citation_valid_rate": 0.8},
        {"prompt_style": "system_examples", "abstention_accuracy": 1.0, "false_refusal_rate": 0.2,
         "first_attempt_citation_valid_rate": 0.9},
    ])
    # system_examples abstains best but refuses too many answerable questions
    assert choose_prompt_style(table, max_false_refusal=0.10) == "system"
    assert choose_prompt_style(table, max_false_refusal=0.25) == "system_examples"
    # tie on abstention -> better first-attempt citations; then the simpler style
    table.loc[2, ["abstention_accuracy", "false_refusal_rate"]] = [0.8, 0.04]
    assert choose_prompt_style(table, 0.10) == "system_examples"
    table.loc[2, "first_attempt_citation_valid_rate"] = 0.8
    assert choose_prompt_style(table, 0.10) == "system"


def test_choose_prompt_style_prefers_fewer_false_refusals_on_equal_abstention():
    import pandas as pd
    from evaluation.eval_abstention import choose_prompt_style
    # the case found on 2026-10-26: same abstention, one style refuses answerable questions
    table = pd.DataFrame([
        {"prompt_style": "single", "abstention_accuracy": 0.2, "false_refusal_rate": 0.00,
         "first_attempt_citation_valid_rate": 0.93},
        {"prompt_style": "system", "abstention_accuracy": 1.0, "false_refusal_rate": 0.08,
         "first_attempt_citation_valid_rate": 0.95},
        {"prompt_style": "system_v2", "abstention_accuracy": 1.0, "false_refusal_rate": 0.04,
         "first_attempt_citation_valid_rate": 0.90},
    ])
    assert choose_prompt_style(table, 0.10) == "system_v2"
    # the real result of the first three styles is unchanged: only "single" was within 10%
    real = pd.DataFrame([
        {"prompt_style": "single", "abstention_accuracy": 0.2, "false_refusal_rate": 0.00,
         "first_attempt_citation_valid_rate": 0.931},
        {"prompt_style": "system", "abstention_accuracy": 1.0, "false_refusal_rate": 0.12,
         "first_attempt_citation_valid_rate": 0.864},
        {"prompt_style": "system_examples", "abstention_accuracy": 1.0, "false_refusal_rate": 0.20,
         "first_attempt_citation_valid_rate": 0.900},
    ])
    assert choose_prompt_style(real, 0.10) == "single"


def test_pipeline_reads_the_chosen_prompt_style():
    import tempfile
    from evaluation.eval_abstention import save_generation_config
    from rag_pipeline import load_prompt_style
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "generation.json"
        assert load_prompt_style(path) is None
        save_generation_config("system_examples", path)
        assert load_prompt_style(path) == "system_examples"


def test_mock_provider_still_works():
    out = guarded_answer(Q005, retrieved_for_q005(), provider="mock")
    assert not out["abstained"] and out["citation_check"]["valid"]


# ======================================================== calibration
def test_calibration_respects_the_false_refusal_budget():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.15, 0.6]
    answerable = [True, True, True, True, False, False, False]

    loose = calibrate_threshold(scores, answerable, max_false_refusal=0.25)
    assert loose["min_rerank_score"] == pytest.approx(0.65)
    assert loose["abstention_accuracy"] == 1.0 and loose["false_refusal_rate"] == 0.25

    strict = calibrate_threshold(scores, answerable, max_false_refusal=0.0)
    assert strict["min_rerank_score"] == pytest.approx(0.175)
    assert strict["false_refusal_rate"] == 0.0


def test_calibration_leaves_threshold_off_when_it_cannot_help():
    # unanswerable "fact" questions (Q032-Q041) can score as high as answerable ones
    out = calibrate_threshold([0.2, 0.9], [True, False], max_false_refusal=0.0)
    assert out["min_rerank_score"] is None


# ======================================================== gold-set loader
def test_gold_article_numbers_stored_as_floats_are_parsed_correctly():
    from evaluation.eval_abstention import _doc_prefix, _parse_articles
    assert _parse_articles(1.0) == ["1"]          # pandas reads 1 as 1.0
    assert _parse_articles("10.0") == ["10"]
    assert _parse_articles("5, 8") == ["5", "8"]
    assert _parse_articles(float("nan")) == []
    assert _doc_prefix("implementing_ar") == "implementing"


# ======================================================== Gemini provider
def test_gemini_provider_sends_the_prompt_and_reads_the_text(monkeypatch):
    """No network: a fake `google.genai` module stands in for the SDK."""
    import types

    from generation.generation import call_llm

    sent = {}

    class FakeInteractions:
        def create(self, model, input, generation_config):
            sent.update(model=model, input=input)
            return types.SimpleNamespace(output_text=Q005_GOOD)

    class FakeClient:
        interactions = FakeInteractions()

    genai = types.ModuleType("google.genai")
    genai.Client = FakeClient
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert call_llm("PROMPT", provider="gemini") == Q005_GOOD
    assert sent == {"model": "gemini-3.8-flash", "input": "PROMPT"}


def _install_fake_gemini(monkeypatch, behaviour):
    """behaviour(model) returns text or raises, like the real SDK would."""
    import types

    calls = []

    class FakeInteractions:
        def create(self, model, input, generation_config):
            calls.append(model)
            return types.SimpleNamespace(output_text=behaviour(model))

    class FakeClient:
        interactions = FakeInteractions()

    genai = types.ModuleType("google.genai")
    genai.Client = FakeClient
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_RETRIES", "2")
    return calls


HIGH_DEMAND = ("Error code: 503 - {'error': {'message': 'gemini-3.8-flash is currently experiencing "
               "high demand, spikes in demand are usually temporary. Please try again later.', "
               "'code': 'service_unavailable'}}")


def test_gemini_503_is_retried_then_falls_back_to_the_lighter_model(monkeypatch):
    from generation.generation import LAST_LLM, call_llm

    def behaviour(model):
        if model == "gemini-3.8-flash":
            raise RuntimeError(HIGH_DEMAND)  # the exact error seen on 2026-10-21
        return Q005_GOOD

    calls = _install_fake_gemini(monkeypatch, behaviour)
    assert call_llm("PROMPT", provider="gemini") == Q005_GOOD
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert LAST_LLM["model"] == "gemini-3.5-flash-lite"


def test_gemini_recovers_when_the_spike_passes(monkeypatch):
    from generation.generation import LAST_LLM, call_llm

    state = {"n": 0}

    def behaviour(model):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError(HIGH_DEMAND)
        return Q005_GOOD

    calls = _install_fake_gemini(monkeypatch, behaviour)
    assert call_llm("PROMPT", provider="gemini") == Q005_GOOD
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash"] and LAST_LLM["model"] == "gemini-3.8-flash"


def test_gemini_wrong_api_key_is_not_retried(monkeypatch):
    from generation.generation import call_llm

    def behaviour(model):
        raise RuntimeError("400 API key not valid. Please pass a valid API key.")

    calls = _install_fake_gemini(monkeypatch, behaviour)
    with pytest.raises(RuntimeError):
        call_llm("PROMPT", provider="gemini")
    assert calls == ["gemini-3.8-flash"]  # fails fast: retrying cannot fix a bad key


def test_gemini_all_models_down_gives_a_clear_error(monkeypatch):
    from generation.generation import call_llm

    def behaviour(model):
        raise RuntimeError(HIGH_DEMAND)

    calls = _install_fake_gemini(monkeypatch, behaviour)
    with pytest.raises(RuntimeError):
        call_llm("PROMPT", provider="gemini")
    assert len(calls) == 4  # 2 attempts x 2 models


def test_ollama_500_is_retried_once_and_shows_ollamas_message(monkeypatch):
    import types
    from generation.generation import call_llm

    replies = iter([
        types.SimpleNamespace(status_code=500, text='{"error":"llama runner process has terminated"}'),
        types.SimpleNamespace(status_code=200, text="", json=lambda: {"response": Q005_GOOD,
                                                                       "prompt_eval_count": 900}),
    ])
    monkeypatch.setattr("requests.post", lambda url, json, timeout: next(replies))
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert call_llm("USER", provider="ollama") == Q005_GOOD

    always_500 = types.SimpleNamespace(status_code=500, text='{"error":"CUDA error: out of memory"}')
    monkeypatch.setattr("requests.post", lambda url, json, timeout: always_500)
    with pytest.raises(RuntimeError) as err:
        call_llm("USER", provider="ollama")
    assert "out of memory" in str(err.value)  # the real cause reaches the results file


# ============================================ system_v2 (calibration error analysis)
Q047 = ("شركة علمت بتسرب قد يضر بأصحاب البيانات، وأبلغت الجهة المختصة بعد خمسة أيام دون مبرر. "
        "ما المهلة الواردة في اللائحة؟")


def test_system_v2_tells_the_model_to_apply_rules_to_scenarios():
    from generation.generation import ABSTAIN_REMINDER, RULES, build_messages
    docs = [article(24, IMPL_24_AR, source="implementing")]
    v2 = build_messages(Q047, docs, style="system_v2")
    assert v2["system"] == RULES                        # same rules as "system"
    assert "applying the" in v2["user"] and "statistics" in v2["user"]
    assert ABSTAIN_REMINDER not in v2["user"]           # the literal "specific fact" wording is gone
    assert v2["user"].rstrip().endswith("[Source n] (Article x).")


def test_system_v2_without_abstain_instruction_keeps_the_scenario_rule_only():
    from generation.generation import build_messages, prompt_text
    docs = [article(24, IMPL_24_AR, source="implementing")]
    text = prompt_text(build_messages(Q047, docs, style="system_v2", abstain_instruction=False))
    assert "applying the" in text and "NOT_COVERED" not in text


# ============================================ resumable evaluation (Colab disconnects)
class FakePipe:
    """Stands in for RAGPipeline.ask; can fail once on chosen questions."""

    def __init__(self, fail_once=()):
        self.fail_once = set(fail_once)
        self.calls = []

    def ask(self, question, provider=None, retrieved=None, **kwargs):
        self.calls.append((question, kwargs.get("prompt_style"), kwargs.get("abstain_instruction")))
        if question in self.fail_once:
            self.fail_once.discard(question)
            raise RuntimeError("Ollama error 500: llama runner process has terminated")
        unanswerable = question.startswith("U")
        doc = article(5, PDPL_5_AR, score=0.9)
        return {
            "max_rerank_score": 0.9, "threshold": None, "final_sources": [doc],
            "abstained": unanswerable, "abstention_reason": "llm_not_covered" if unanswerable else None,
            "first_attempt_valid": None if unanswerable else True, "n_attempts": 1,
            "citation_check": None if unanswerable else check_citations(Q005_GOOD, [doc, doc]),
            "answer": "x", "raw_answer": "x", "timings": {"total_s": 0.1},
            "llm_model": "command-r7b-arabic", "prompt_style": kwargs.get("prompt_style"),
        }


def _gold_df():
    import pandas as pd
    return pd.DataFrame({
        "qid": ["Q1", "Q2", "Q3"], "question": ["A1", "A2", "U3"], "language": ["ar", "ar", "en"],
        "answerable": [True, True, False], "gold_articles": [["5"], ["6"], []],
        "gold_source": ["pdpl", "pdpl", ""],
    })


def test_resumable_run_continues_after_a_failure_without_repeating_answers():
    import tempfile
    from evaluation.eval_abstention import run_eval_resumable, summarize
    df = _gold_df()
    cache = {q: {"final": []} for q in df["qid"]}
    pipe = FakePipe(fail_once={"A2"})
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.csv"
        with pytest.raises(RuntimeError) as err:
            run_eval_resumable(pipe, df, cache, ckpt, label="final", verbose=False)
        assert "Q2" in str(err.value) and "llama runner" in str(err.value)
        rows = run_eval_resumable(pipe, df, cache, ckpt, label="final", verbose=False)

    asked = [c[0] for c in pipe.calls]
    assert asked == ["A1", "A2", "A2", "U3"]           # A1 was not asked again
    assert list(rows["qid"]) == ["Q1", "Q2", "Q3"]
    s = summarize(rows)                                # types survive the CSV round-trip
    assert s["abstention_accuracy"] == 1.0 and s["false_refusal_rate"] == 0.0
    assert s["abstention_reasons"] == {"llm_not_covered": 1}


def test_compare_prompt_styles_reuses_styles_already_in_the_checkpoint():
    import tempfile
    from evaluation.eval_abstention import compare_prompt_styles, run_eval_resumable
    df = _gold_df()
    cache = {q: {"final": []} for q in df["qid"]}
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "styles.csv"
        earlier = FakePipe()
        run_eval_resumable(earlier, df, cache, ckpt, label="style_single", prompt_style="single",
                           verbose=False)
        later = FakePipe()
        table, rows, chosen = compare_prompt_styles(later, df, cache, "ollama",
                                                    styles=("single", "system_v2"), checkpoint=ckpt)
    assert {c[1] for c in later.calls} == {"system_v2"}   # "single" came from the file
    assert list(table["prompt_style"]) == ["single", "system_v2"]
    assert list(table["n_questions"]) == [3, 3]
    assert chosen == "single"                              # tie -> the simpler, earlier style