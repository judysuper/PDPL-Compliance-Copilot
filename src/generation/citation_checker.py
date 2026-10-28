"""Citation checker: verify that the articles an answer cites were actually retrieved.

What is checked automatically
- the answer contains citations in the required format "[Source n] (Article x)";
- every [Source n] tag points to a source that was really sent to the LLM;
- every article number mentioned in the answer belongs to a retrieved article;
- every "[Source n] (Article x)" pair agrees (source n really is article x).

Cross-references: retrieved articles often point to other articles (Article 6 says
"... paragraph (1) of Article (5) ..."). If the answer repeats such a reference, the
number was not invented, so by default it is reported as a warning
(`cross_referenced_articles`) and not as an unsupported citation. The text of that
other article was not retrieved, so these answers are flagged for human review.

What is NOT checked
- whether the cited article actually SUPPORTS the claim. For that we only give
  a lexical-overlap hint per cited sentence, so reviewers know where to look
  first. A human still has to judge support.
"""

import re

# ---------------------------------------------------------------- normalisation
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹０１２３４５６７８９", "012345678901234567890123456789")


def to_ascii_digits(text: str) -> str:
    """Convert Arabic-Indic, Persian and full-width digits to ASCII digits."""
    return str(text).translate(_DIGITS)


def normalize_text(text: str) -> str:
    """Lower-case, ASCII digits, and light Arabic normalisation (alef, yaa, taa marbuta, diacritics)."""
    text = to_ascii_digits(text).lower()
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


def normalize_article_number(value) -> str:
    """'5.0' / '٥' / ' 5 ' -> '5'."""
    return re.sub(r"\.0+$", "", to_ascii_digits(str(value)).strip())


def _num_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


# --------------------------------------------- Arabic ordinals: "المادة الخامسة"
_UNITS = {1: "الحاديه", 2: "الثانيه", 3: "الثالثه", 4: "الرابعه", 5: "الخامسه",
          6: "السادسه", 7: "السابعه", 8: "الثامنه", 9: "التاسعه"}
_TENS = {2: ("العشرون", "العشرين"), 3: ("الثلاثون", "الثلاثين"),
         4: ("الاربعون", "الاربعين"), 5: ("الخمسون", "الخمسين"),
         6: ("الستون", "الستين")}


def _build_ordinals():
    """Map normalised ordinal words (1-69) to numbers, e.g. 'الحاديه والعشرون' -> 21."""
    forms = {"الاولي": 1, "العاشره": 10}
    for u, word in _UNITS.items():
        if u > 1:
            forms[word] = u
        forms[f"{word} عشره"] = 10 + u
        forms[f"{word} عشر"] = 10 + u
    for t, tens_words in _TENS.items():
        for tens in tens_words:
            forms[tens] = t * 10
            for u, word in _UNITS.items():
                forms[f"{word} و{tens}"] = t * 10 + u
    return forms


_ORDINALS = _build_ordinals()


def _ordinal_regex(form: str) -> str:
    parts = []
    for word in form.split(" "):
        if word.startswith("و"):
            parts.append(r"و\s*" + re.escape(word[1:]))
        else:
            parts.append(re.escape(word))
    return r"\s+".join(parts)


_ORD_PATTERN = "|".join(
    _ordinal_regex(f) for f in sorted(_ORDINALS, key=len, reverse=True)
)


def _canonical_ordinal(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r" و\s*", " و", text)


# -------------------------------------------------------------- citation regex
_NUM = r"\(?\s*\d+\s*\)?"
_NUM_LIST = _NUM + r"(?:\s*(?:,|،|and|&|و)\s*" + _NUM + r")*"

# Patterns run on normalize_text() output, so Arabic keywords are written normalised:
# "ماده" matches المادة / للمادة / بالمادة, "مواد" matches المواد.
ARTICLE_RE = re.compile(
    r"(?:\barticles?\b|\bart\.|ماده|مواد)\s*(?:no\.?|رقم)?\s*[:\-]?\s*"
    r"(?P<ref>" + _NUM_LIST + r"|\(?\s*(?:" + _ORD_PATTERN + r")\s*\)?)"
)
SOURCE_RE = re.compile(r"(?:\bsource\b|المصدر|مصدر)\s*#?\s*(\d+)")

# How far after a [Source n] tag an article reference may start to count as its pair.
PAIR_WINDOW = 15


def _article_numbers(ref: str):
    numbers = re.findall(r"\d+", ref)
    if numbers:
        return numbers
    words = _canonical_ordinal(ref.strip("() \t"))
    return [str(_ORDINALS[words])] if words in _ORDINALS else []


def extract_articles(text: str):
    """All article numbers mentioned in a text (digits, Arabic digits, lists, Arabic ordinals)."""
    found = []
    for m in ARTICLE_RE.finditer(normalize_text(text)):
        found.extend(_article_numbers(m.group("ref")))
    return found


def extract_sources(text: str):
    return [int(n) for n in SOURCE_RE.findall(normalize_text(text))]


def extract_pairs(text: str):
    """[(source_number, [article numbers]), ...] for tags written as [Source n] (Article x)."""
    norm = normalize_text(text)
    matches = list(SOURCE_RE.finditer(norm))
    pairs = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(norm)
        window = norm[m.end():min(end, m.end() + PAIR_WINDOW + 40)]
        art = ARTICLE_RE.search(window)
        if art and art.start() <= PAIR_WINDOW:
            pairs.append((int(m.group(1)), _article_numbers(art.group("ref"))))
    return pairs


# ------------------------------------------------- support hint (lexical overlap)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "may", "shall",
    "not", "any", "its", "their", "has", "have", "which", "such", "under", "into",
    "source", "article", "articles",
    "في", "من", "علي", "الي", "عن", "ان", "او", "التي", "الذي", "هذا", "هذه",
    "ذلك", "مع", "كل", "ما", "لا", "اذا", "وفق", "وفقا", "ماده", "الماده",
    "المصدر", "مصدر", "حسب", "يجب", "يحق",
}
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و")


def _tokens(text: str):
    tokens = set()
    for word in re.findall(r"[a-zء-ي]{3,}", normalize_text(text)):
        if word in _STOPWORDS:
            continue
        for p in _PREFIXES:
            if word.startswith(p) and len(word) - len(p) >= 3:
                word = word[len(p):]
                break
        tokens.add(word)
    return tokens


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", text or ""))


def _split_sentences(text: str):
    parts = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def claim_overlaps(answer: str, docs):
    """For each sentence with [Source n] tags: share of its words found in the cited articles.

    A review hint only. A high overlap does not prove support and a low overlap
    does not prove a wrong citation. Returns None when the sentence and the
    cited article are in different languages (overlap is meaningless there).
    """
    claims = []
    for sentence in _split_sentences(answer):
        sources = sorted({n for n in extract_sources(sentence) if 1 <= n <= len(docs)})
        if not sources:
            continue
        cited = [docs[n - 1] for n in sources]
        doc_text = " ".join(
            f"{d.get('article_title') or ''} {d.get('article_text') or ''}" for d in cited
        )
        if _is_arabic(sentence) != _is_arabic(doc_text):
            overlap = None
        else:
            words = _tokens(sentence)
            overlap = round(len(words & _tokens(doc_text)) / len(words), 3) if words else None
        claims.append({"sentence": sentence, "sources": sources, "overlap": overlap})
    return claims


# ------------------------------------------------------------------ main check
def check_citations(answer: str, retrieved_chunks, require_source_tags: bool = True,
                    allow_cross_references: bool = True) -> dict:
    """Check the citations in `answer` against the chunks that were sent to the LLM.

    retrieved_chunks: list of dicts with at least `article_number` (and ideally
    `article_text`), in the same order as the [Source n] blocks of the prompt.
    """
    docs = list(retrieved_chunks or [])
    allowed = {normalize_article_number(d.get("article_number", "")) for d in docs}
    allowed.discard("")

    cited_sources = sorted(set(extract_sources(answer)))
    invalid_sources = [n for n in cited_sources if not 1 <= n <= len(docs)]

    cited_articles = sorted(set(extract_articles(answer)), key=_num_key)
    supported = [a for a in cited_articles if a in allowed]
    not_retrieved = [a for a in cited_articles if a not in allowed]

    referenced = set()
    for d in docs:
        referenced.update(extract_articles(d.get("article_text") or ""))
    if allow_cross_references:
        cross_referenced = [a for a in not_retrieved if a in referenced]
        unsupported = [a for a in not_retrieved if a not in referenced]
    else:
        cross_referenced, unsupported = [], not_retrieved

    mismatched = []
    for source, articles in extract_pairs(answer):
        if not 1 <= source <= len(docs) or not articles:
            continue  # out-of-range sources are already reported above
        actual = normalize_article_number(docs[source - 1].get("article_number", ""))
        if actual not in articles:
            mismatched.append({"source": source, "cited_article": articles[0],
                               "actual_article": actual})

    has_citations = bool(cited_sources) if require_source_tags else bool(cited_sources or cited_articles)

    issues = []
    if not has_citations:
        issues.append("no citations in the required format [Source n] (Article x)")
    if invalid_sources:
        issues.append(f"source tags not sent to the model: {invalid_sources}")
    if unsupported:
        issues.append(f"articles not in the retrieved sources: {unsupported}")
    if mismatched:
        issues.append("source/article pairs that do not match: " + ", ".join(
            f"[Source {m['source']}] is Article {m['actual_article']}, not {m['cited_article']}"
            for m in mismatched))

    warnings = []
    if cross_referenced:
        warnings.append("articles only cross-referenced inside the retrieved text "
                        f"(their own text was not retrieved): {cross_referenced}")

    claims = claim_overlaps(answer, docs)
    overlaps = [c["overlap"] for c in claims if c["overlap"] is not None]

    valid = not issues
    return {
        "valid": valid,
        "all_valid": valid,  # same key as day-2 check, so old code keeps working
        "has_citations": has_citations,
        "cited_sources": cited_sources,
        "invalid_sources": invalid_sources,
        "cited_articles": cited_articles,
        "supported_articles": supported,
        "unsupported_articles": unsupported,
        "cross_referenced_articles": cross_referenced,
        "mismatched_pairs": mismatched,
        "claims": claims,
        "min_claim_overlap": min(overlaps) if overlaps else None,
        "issues": issues,
        "warnings": warnings,
    }