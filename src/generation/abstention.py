"""Abstention: refuse to answer when the evidence is not good enough.

The policy has two gates around the LLM call:

  before generation  - no retrieved articles            -> abstain ("no_documents")
                     - best rerank score < threshold    -> abstain ("low_retrieval_score")
  after generation   - LLM replied NOT_COVERED:         -> abstain ("llm_not_covered")
                     - citations fail the checker       -> regenerate once with the
                       allowed citations; if still failing -> abstain
                       ("no_citations" / "unsupported_citations")

The threshold is never guessed: it is chosen by `calibrate_threshold` on the
calibration half of the gold Q&A set (see src/evaluation/eval_abstention.py).
"""

import math
import os
import re

from generation.citation_checker import check_citations, normalize_article_number
from generation.generation import (
    LAST_LLM,
    _mock_answer,
    build_messages,
    call_llm,
    is_arabic,
    prompt_text,
)

BASE_MESSAGE = {
    "ar": "لا أستطيع تقديم إجابة موثوقة بناءً على المصادر القانونية المتاحة حاليًا. "
          "يرجى توفير مصدر ذي صلة أو مراجعة النص النظامي الرسمي.",
    "en": "I cannot provide a reliable answer based on the legal sources currently available. "
          "Please provide a relevant source or consult the official regulatory text.",
}

REASON_TEXT = {
    "no_documents": {
        "ar": "لم تُسترجع أي مواد قانونية للسؤال.",
        "en": "No legal articles were retrieved for this question.",
    },
    "low_retrieval_score": {
        "ar": "المواد المسترجعة ليست وثيقة الصلة بالسؤال بدرجة كافية.",
        "en": "The retrieved articles are not relevant enough to the question.",
    },
    "llm_not_covered": {
        "ar": "المصادر المسترجعة لا تتضمن ما يكفي للإجابة.",
        "en": "The retrieved sources do not contain enough to answer.",
    },
    "no_citations": {
        "ar": "الإجابة المولّدة لم تستند إلى أي مادة مسترجعة.",
        "en": "The generated answer did not cite any retrieved article.",
    },
    "unsupported_citations": {
        "ar": "الإجابة المولّدة استشهدت بمواد غير موجودة في المصادر المسترجعة.",
        "en": "The generated answer cited articles that are not in the retrieved sources.",
    },
}

NOT_COVERED_RE = re.compile(r"^\W*NOT_COVERED\W*", re.IGNORECASE)


def is_not_covered(answer: str) -> bool:
    """True if the LLM abstained, also when it wrapped the tag in markdown (**NOT_COVERED:**)."""
    return bool(NOT_COVERED_RE.match((answer or "").strip()))


def abstention_message(reason: str, lang: str = "ar", detail: str = None) -> str:
    lang = lang if lang in BASE_MESSAGE else "en"
    text = BASE_MESSAGE[lang]
    if reason in REASON_TEXT:
        text += "\n" + REASON_TEXT[reason][lang]
    if detail:
        text += "\n\n" + detail.strip()
    return text


# ----------------------------------------------------------- gate 1: evidence
def evidence_gate(docs, min_score=None, score_key="rerank_score"):
    """Return (kept_docs, reason_or_None, best_score).

    With a threshold, articles scoring below it are dropped; if none remain the
    system abstains. Without a threshold only the empty case abstains.
    """
    docs = list(docs or [])
    scores = [d[score_key] for d in docs if d.get(score_key) is not None]
    best = max(scores) if scores else None

    if not docs:
        return [], "no_documents", best

    if min_score is None:
        return docs, None, best

    if len(scores) != len(docs):
        raise ValueError(
            f"A threshold was given but some articles have no '{score_key}'. "
            "Thresholds are calibrated on reranker scores, so run with the reranker."
        )

    kept = [d for d in docs if d[score_key] >= min_score]
    return kept, (None if kept else "low_retrieval_score"), best


# ----------------------------------------------------------------- prompting
def build_answer_prompt(question, docs, abstain_instruction=True, prompt_style=None):
    """{"system", "user"} messages for the chosen prompt style (see generation.build_messages).

    abstain_instruction=False removes every NOT_COVERED instruction (experiment E5).
    """
    return build_messages(question, docs, style=prompt_style, abstain_instruction=abstain_instruction)


def repair_prompt(base_messages, report, docs, abstain_instruction=True):
    """Ask the LLM to rewrite an answer whose citations failed the checker."""
    allowed = ", ".join(
        f"[Source {i}] (Article {normalize_article_number(d.get('article_number', ''))})"
        for i, d in enumerate(docs, start=1)
    )
    note = (
        "IMPORTANT: your previous answer was rejected by an automatic citation check "
        f"({'; '.join(report['issues'])}).\n"
        f"Rewrite the answer citing only these sources: {allowed}.\n"
        "Use exactly the format [Source n] (Article x) and do not mention any other article number."
    )
    if abstain_instruction:
        note += "\nIf these sources are not sufficient, begin your reply with NOT_COVERED:"
    return {"system": base_messages.get("system"), "user": f"{base_messages['user']}\n\n{note}"}


# --------------------------------------------------------- the guarded answer
def guarded_answer(
    question,
    docs,
    provider=None,
    language=None,
    min_rerank_score=None,
    score_key="rerank_score",
    abstain_instruction=True,
    enforce_citations=True,
    max_retries=1,
    llm_fn=None,
    prompt_style=None,
):
    """Generate an answer only if the evidence and the citations pass both gates.

    llm_fn: optional callable(prompt_text) -> str, used by the tests instead of a real LLM.
    prompt_style: "single" | "system" | "system_examples" (default: env PROMPT_STYLE or "system").
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    lang = language or ("ar" if is_arabic(question) else "en")

    kept, reason, best = evidence_gate(docs, min_rerank_score, score_key)
    result = {
        "language": lang,
        "provider": provider,
        "threshold": min_rerank_score,
        "max_score": best,
        "sources_used": kept,
        "attempts": [],
        "first_attempt_valid": None,
        "citation_check": None,
        "raw_answer": None,
        "llm_model": None,
        "prompt_style": None,
    }

    def finish(answer, abstained, why=None):
        result.update(answer=answer, abstained=abstained, abstention_reason=why,
                      n_attempts=len(result["attempts"]))
        return result

    if reason:
        result["sources_used"] = []
        return finish(abstention_message(reason, lang), True, reason)

    prompt = build_answer_prompt(question, kept, abstain_instruction, prompt_style)
    result["prompt_style"] = prompt_style or os.getenv("PROMPT_STYLE") or "system"

    def generate(messages):
        if llm_fn is not None:
            return llm_fn(prompt_text(messages))
        if provider == "mock":
            return _mock_answer(question, kept)
        text = call_llm(messages["user"], provider, system=messages.get("system"))
        result["llm_model"] = LAST_LLM.get("model")  # may be a fallback model
        return text

    can_retry = llm_fn is not None or provider != "mock"
    answer = generate(prompt)

    for attempt in range(max_retries + 1):
        result["raw_answer"] = answer

        if is_not_covered(answer):
            result["attempts"].append({"answer": answer, "valid": None, "issues": ["NOT_COVERED"]})
            detail = NOT_COVERED_RE.sub("", answer.strip(), count=1)
            return finish(abstention_message("llm_not_covered", lang, detail), True, "llm_not_covered")

        report = check_citations(answer, kept)
        result["citation_check"] = report
        result["attempts"].append({"answer": answer, "valid": report["valid"], "issues": report["issues"]})
        if result["first_attempt_valid"] is None:
            result["first_attempt_valid"] = report["valid"]

        if report["valid"] or not enforce_citations:
            return finish(answer, False)

        if attempt < max_retries and can_retry:
            answer = generate(repair_prompt(prompt, report, kept, abstain_instruction))
        else:
            break

    why = "no_citations" if not report["has_citations"] else "unsupported_citations"
    return finish(abstention_message(why, lang), True, why)


# ------------------------------------------------------- threshold calibration
def calibrate_threshold(max_scores, answerable, max_false_refusal=0.10):
    """Pick the retrieval-score threshold from labelled calibration questions.

    max_scores: best rerank score per question; answerable: True/False per question.
    A question is refused when its best score < threshold. We choose the threshold
    that refuses the most unanswerable questions while refusing at most
    `max_false_refusal` of the answerable ones (ties -> the lower threshold).
    Returns a dict with the chosen threshold (None = do not use a threshold) and the
    full curve for the report.
    """
    pairs = [(-math.inf if s is None else float(s), bool(a)) for s, a in zip(max_scores, answerable)]
    pos = [s for s, a in pairs if a]
    neg = [s for s, a in pairs if not a]
    if not pos or not neg:
        raise ValueError("Calibration needs both answerable and unanswerable questions.")

    values = sorted({s for s, _ in pairs if s != -math.inf})
    candidates = [None] + [(a + b) / 2 for a, b in zip(values, values[1:])]

    curve = []
    for t in candidates:
        refused_pos = sum(s < t for s in pos) if t is not None else 0
        refused_neg = sum(s < t for s in neg) if t is not None else 0
        curve.append({
            "threshold": t,
            "abstention_accuracy": refused_neg / len(neg),
            "false_refusal_rate": refused_pos / len(pos),
        })

    allowed = [r for r in curve if r["false_refusal_rate"] <= max_false_refusal]
    best = max(allowed, key=lambda r: (r["abstention_accuracy"],
                                       -(r["threshold"] if r["threshold"] is not None else -math.inf)))
    if best["abstention_accuracy"] == 0:
        best = curve[0]  # a threshold does not help: leave it off

    return {
        "min_rerank_score": best["threshold"],
        "abstention_accuracy": best["abstention_accuracy"],
        "false_refusal_rate": best["false_refusal_rate"],
        "max_false_refusal": max_false_refusal,
        "n_answerable": len(pos),
        "n_unanswerable": len(neg),
        "curve": curve,
    }