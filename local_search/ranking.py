import math
from typing import Dict, Optional, Protocol, Sequence

from .analysis import field_tokens
from .matcher import ClauseHit
from .query import Query
from .segment import StoredDocument


class Ranker(Protocol):
    def score(
        self,
        documents: Sequence[StoredDocument],
        query: Query,
    ) -> Dict[str, float]:
        ...


class ConstantScoreRanker:
    def score(
        self,
        documents: Sequence[StoredDocument],
        query: Query,
    ) -> Dict[str, float]:
        return {document.id: 0.0 for document in documents}

    def explain(
        self,
        documents: Sequence[StoredDocument],
        query: Query,
        hits: Dict[str, Sequence[ClauseHit]],
    ) -> Dict[str, object]:
        return {
            document.id: {
                "total": 0.0,
                "terms": [
                    {
                        "term": term.term,
                        "match": term.kind,
                        "matched": term.matched_token,
                        "tf": len(term.positions),
                        "idf": 0.0,
                        "contribution": 0.0,
                    }
                    for clause in hits.get(document.id, [])
                    for term in clause.terms
                ],
            }
            for document in documents
        }


class BM25Ranker:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        documents: Sequence[StoredDocument],
        query: Query,
    ) -> Dict[str, float]:
        explanation = self.explain(documents, query, {})
        return {
            document.id: float(explanation.get(document.id, {}).get("total", 0.0))
            for document in documents
        }

    def explain(
        self,
        documents: Sequence[StoredDocument],
        query: Query,
        hits: Optional[Dict[str, Sequence[ClauseHit]]] = None,
    ) -> Dict[str, object]:
        document_count = len(documents) or 1
        terms = list(dict.fromkeys(term for clause in query.clauses for term in clause.terms))
        token_fields = [field_tokens(doc.fields) for doc in documents]
        lengths = [sum(len(tokens) for tokens in fields.values()) for fields in token_fields]
        average_length = sum(lengths) / document_count or 1.0

        hit_kind: Dict[str, Dict[str, tuple[str, str]]] = {}
        for doc in documents:
            per_term: Dict[str, tuple[str, str]] = {}
            for clause in (hits or {}).get(doc.id, []):
                for term in clause.terms:
                    per_term.setdefault(term.term, (term.kind, term.matched_token))
            hit_kind[doc.id] = per_term

        counted_tokens: Dict[str, Dict[str, frozenset[str]]] = {}
        for doc, fields in zip(documents, token_fields):
            per_term_tokens: Dict[str, frozenset[str]] = {}
            all_tokens = [token for tokens in fields.values() for token in tokens]
            for term in terms:
                kind_info = hit_kind.get(doc.id, {}).get(term)
                if kind_info is None:
                    continue
                kind, matched = kind_info
                if kind == "synonym" and " " in matched:
                    words = matched.split(" ")
                    found = set()
                    for start in range(len(all_tokens) - len(words) + 1):
                        if all_tokens[start : start + len(words)] == words:
                            found.update(words)
                    per_term_tokens[term] = frozenset(found)
                else:
                    per_term_tokens[term] = frozenset({matched})
            counted_tokens[doc.id] = per_term_tokens

        result: Dict[str, object] = {}
        for doc, fields, length in zip(documents, token_fields, lengths):
            term_details = []
            total = 0.0
            for term in terms:
                counted = counted_tokens.get(doc.id, {}).get(term)
                if counted is not None:
                    frequency = sum(
                        tokens.count(token)
                        for tokens in fields.values()
                        for token in counted
                    )
                else:
                    frequency = sum(tokens.count(term) for tokens in fields.values())
                if not frequency:
                    continue
                if counted is not None:
                    containing = sum(
                        1
                        for other_fields in token_fields
                        if any(any(token in counted for token in tokens) for tokens in other_fields.values())
                    )
                else:
                    containing = sum(
                        1
                        for other_fields in token_fields
                        if any(term in tokens for tokens in other_fields.values())
                    )
                idf = math.log(1 + (document_count - containing + 0.5) / (containing + 0.5))
                normalization = self.k1 * (1 - self.b + self.b * length / average_length)
                contribution = idf * frequency * (self.k1 + 1) / (frequency + normalization)
                total += contribution
                kind_info = hit_kind.get(doc.id, {}).get(term, ("original", term))
                term_details.append(
                    {
                        "term": term,
                        "match": kind_info[0],
                        "matched": kind_info[1],
                        "tf": frequency,
                        "idf": idf,
                        "contribution": contribution,
                    }
                )
            result[doc.id] = {"total": total, "terms": term_details}
        return result
