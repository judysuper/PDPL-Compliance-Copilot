from __future__ import annotations
import re
import unicodedata


# OPP-115 categories -> PDPL requirements. OPP is supporting evidence only;
# requirements such as breach notification, ROPA, DPO and DPIA are primarily
# decided by the PDPL-specific rule layer.
OPP_CATEGORY_MAP = {
    "REQ-001": ["First Party Collection/Use"],
    "REQ-002": ["User Choice/Control", "First Party Collection/Use"],
    "REQ-003": ["User Choice/Control"],
    "REQ-004": ["First Party Collection/Use"],
    "REQ-005": ["User Access, Edit and Deletion"],
    "REQ-006": ["User Access, Edit and Deletion"],
    "REQ-007": ["User Access, Edit and Deletion", "Data Retention"],
    "REQ-008": ["Data Security"],
    "REQ-009": ["Data Security"],
    "REQ-010": ["International and Specific Audiences"],
    "REQ-011": ["First Party Collection/Use", "Other"],
    "REQ-012": ["Other"],
    "REQ-013": ["Other"],
    "REQ-014": ["Other"],
    "REQ-015": ["Third Party Sharing/Collection"],
}


# Each requirement is expressed as keyword groups.
# Covered normally requires enough groups, Partial means incomplete evidence,
# Missing means no meaningful notice-level evidence.
RULES = {
    "REQ-001": {
        "groups": [
            ["غرض", "أغراض", "purpose", "purposes", "why we collect", "use your personal"],
            ["مسوغ", "أساس نظامي", "أساس قانوني", "legal basis", "lawful basis", "contract", "consent", "legitimate interest"],
        ],
        "covered_groups": 2,
    },
    "REQ-002": {
        "groups": [
            ["موافقة", "consent"],
            ["سحب الموافقة", "withdraw consent", "withdrawal of consent"],
        ],
        "covered_groups": 2,
    },
    "REQ-003": {
        "groups": [
            ["شرط", "مشروط", "condition", "conditional"],
            ["الخدمة", "منفعة", "service", "benefit"],
            ["غير ضروري", "غير مرتبط", "not necessary", "unrelated", "not required"],
        ],
        "covered_groups": 3,
    },
    "REQ-004": {
        "groups": [
            ["نجمع", "جمع البيانات", "نحصل على", "collect", "collection", "obtain"],
            ["منك مباشرة", "مباشرة منك", "directly from you", "from you"],
            ["الغرض", "أغراض", "purpose", "purposes"],
        ],
        "covered_groups": 2,
    },
    "REQ-005": {
        "groups": [
            ["الوصول", "الاطلاع", "نسخة", "access", "copy of your data", "obtain a copy"],
        ],
        "covered_groups": 1,
    },
    "REQ-006": {
        "groups": [
            ["تصحيح", "تحديث", "إتمام", "rectif", "correct", "update", "complete"],
        ],
        "covered_groups": 1,
    },
    "REQ-007": {
        "groups": [
            ["إتلاف", "حذف", "محو", "delete", "deletion", "erase", "erasure", "destroy"],
            ["احتفاظ", "مدة الاحتفاظ", "retention", "retain"],
        ],
        "covered_groups": 2,
    },
    "REQ-008": {
        "groups": [
            ["أمن", "حماية", "تشفير", "ضوابط", "security", "protect", "encryption", "technical measures", "organizational measures"],
            ["وصول غير مصرح", "تسريب", "اختراق", "unauthorised access", "unauthorized access", "breach"],
        ],
        "covered_groups": 1,
    },
    "REQ-009": {
        "groups": [
            ["خرق", "اختراق", "تسريب", "breach", "incident"],
            ["إبلاغ", "إشعار", "إخطار", "notify", "notification", "report"],
            ["الجهة المختصة", "سدايا", "أصحاب البيانات", "affected", "competent authority", "regulator", "data subject"],
        ],
        "covered_groups": 2,
    },
    "REQ-010": {
        "groups": [
            ["خارج المملكة", "خارج السعودية", "outside the kingdom", "outside saudi", "international transfer", "cross-border", "overseas"],
            ["ضمان", "مستوى مناسب", "adequate protection", "safeguard", "standard contractual", "transfer regulation"],
        ],
        "covered_groups": 1,
    },
    "REQ-011": {
        "groups": [
            ["بيانات حساسة", "بيانات صحية", "بيانات حيوية", "بيانات ائتمانية", "sensitive data", "health data", "biometric", "credit data"],
            ["موافقة صريحة", "explicit consent", "express consent"],
        ],
        "covered_groups": 2,
    },
    "REQ-012": {
        "groups": [
            ["سجل عمليات المعالجة", "سجل أنشطة المعالجة", "ropa", "record of processing", "records of processing"],
        ],
        "covered_groups": 1,
        "strict": True,
    },
    "REQ-013": {
        "groups": [
            ["مسؤول حماية البيانات", "مسؤول الخصوصية", "data protection officer", "dpo", "privacy officer"],
            ["تواصل", "بريد", "email", "contact"],
        ],
        "covered_groups": 1,
    },
    "REQ-014": {
        "groups": [
            ["تقييم أثر حماية البيانات", "تقييم الأثر", "dpia", "data protection impact assessment", "privacy impact assessment"],
        ],
        "covered_groups": 1,
        "strict": True,
    },
    "REQ-015": {
        "groups": [
            ["طرف ثالث", "أطراف ثالثة", "نشارك", "نفصح", "third party", "third parties", "share", "disclose", "disclosure"],
            ["موافقة", "مسوغ", "قانون", "consent", "law", "legal", "required by"],
        ],
        "covered_groups": 1,
    },
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    # Arabic normalization, light only.
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def group_matches(text: str, groups: list[list[str]]) -> tuple[int, list[str]]:
    ntext = normalize_text(text)
    matched = []
    for group in groups:
        hit = None
        for term in group:
            if normalize_text(term) in ntext:
                hit = term
                break
        if hit:
            matched.append(hit)
    return len(matched), matched
