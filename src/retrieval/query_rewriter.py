from __future__ import annotations


class QueryRewriter:
    def rewrite(self, question: str) -> str:
        q = question.strip()
        if not q:
            return q

        replacements = {
            "booking revenue rules": "What are the business rules for calculating booking revenue?",
            "revenue logic": "What is the business logic for calculating total paid booking revenue?",
        }
        return replacements.get(q.lower(), q)