from __future__ import annotations
from dataclasses import asdict
from typing import Any

from checklist_v2 import Requirement
from policy_parser_v2 import split_paragraphs
from rules_v2 import RULES, OPP_CATEGORY_MAP, group_matches


class GapAnalyzerV2:
    """
    PDPL Gap Analyzer v2:
      - 15 requirements from Member 1 checklist
      - OPP-115 XLM-R labels as supporting paragraph evidence
      - PDPL-specific rule layer for requirements not represented by OPP
      - Output restricted to Covered / Partial / Missing
    """

    def __init__(self, requirements: list[Requirement], classifier=None):
        self.requirements = requirements
        self.classifier = classifier

    def _classifier_evidence(self, paragraphs: list[str], requirement_id: str) -> tuple[float, str, str]:
        if self.classifier is None:
            return 0.0, "", ""

        wanted = OPP_CATEGORY_MAP.get(requirement_id, [])
        if not wanted:
            return 0.0, "", ""

        best_score = 0.0
        best_label = ""
        best_para = ""
        for p in paragraphs:
            try:
                scores = self.classifier.predict_all_scores(p)
            except Exception:
                continue
            for label in wanted:
                s = float(scores.get(label, 0.0))
                if s > best_score:
                    best_score = s
                    best_label = label
                    best_para = p
        return best_score, best_label, best_para

    def _rule_evidence(self, full_text: str, paragraphs: list[str], requirement_id: str):
        cfg = RULES.get(requirement_id, {"groups": [], "covered_groups": 1})
        groups = cfg.get("groups", [])
        full_count, full_terms = group_matches(full_text, groups)

        best_para = ""
        best_para_count = 0
        best_para_terms = []
        for p in paragraphs:
            c, terms = group_matches(p, groups)
            if c > best_para_count:
                best_para_count = c
                best_para_terms = terms
                best_para = p

        return {
            "full_count": full_count,
            "full_terms": full_terms,
            "best_para_count": best_para_count,
            "best_para_terms": best_para_terms,
            "best_para": best_para,
            "covered_groups": int(cfg.get("covered_groups", 1)),
            "strict": bool(cfg.get("strict", False)),
        }

    def _status(self, requirement_id: str, rule_ev: dict[str, Any], cls_score: float) -> str:
        full_count = rule_ev["full_count"]
        needed = rule_ev["covered_groups"]
        strict = rule_ev["strict"]

        # ROPA and DPIA must be explicitly named in a notice to be called Covered.
        if strict:
            if full_count >= needed:
                return "Covered"
            return "Missing"

        if full_count == 0:
            return "Missing"

        # Explicit notice-level evidence can cover requirements even if OPP has no matching category.
        if full_count >= needed:
            if requirement_id in {"REQ-009", "REQ-010", "REQ-011"}:
                # High-risk/legal-specific requirements: classifier is secondary.
                return "Covered"
            if self.classifier is None:
                return "Covered"
            # If OPP category exists and XLM-R supports it, Covered.
            if OPP_CATEGORY_MAP.get(requirement_id) and cls_score >= 0.50:
                return "Covered"
            # Strong PDPL-specific wording but weak OPP signal: conservative Partial.
            return "Partial"

        return "Partial"

    def analyze(self, policy_text: str) -> list[dict[str, Any]]:
        paragraphs = split_paragraphs(policy_text)
        results = []

        for req in self.requirements:
            rule_ev = self._rule_evidence(policy_text, paragraphs, req.requirement_id)
            cls_score, cls_label, cls_para = self._classifier_evidence(
                paragraphs, req.requirement_id
            )
            status = self._status(req.requirement_id, rule_ev, cls_score)

            evidence = rule_ev["best_para"] or cls_para
            results.append({
                "requirement_id": req.requirement_id,
                "requirement": req.requirement,
                "requirement_type": req.requirement_type,
                "source_document": req.source_document,
                "article_number": req.article_number,
                "status": status,
                "matched_terms": "; ".join(rule_ev["full_terms"]),
                "classifier_category": cls_label,
                "classifier_score": round(cls_score, 6),
                "evidence_text": evidence[:1200],
                "verification_question": req.verification_question,
                "legal_reference": req.legal_reference,
            })

        return results
