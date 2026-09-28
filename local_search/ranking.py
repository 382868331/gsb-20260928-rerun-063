import math
from typing import Dict, List, Protocol, Sequence

from .analysis import field_tokens
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


class BM25Ranker:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        documents: Sequence[StoredDocument],
        query: Query,
    ) -> Dict[str, float]:
        document_count = len(documents) or 1
        terms = list(dict.fromkeys(term for clause in query.clauses for term in clause.terms))
        token_fields = [field_tokens(doc.fields) for doc in documents]
        lengths = [sum(len(tokens) for tokens in fields.values()) for fields in token_fields]
        average_length = sum(lengths) / document_count or 1.0
        result = {doc.id: 0.0 for doc in documents}

        for term in terms:
            containing = sum(
                1 for fields in token_fields if any(term in tokens for tokens in fields.values())
            )
            idf = math.log(1 + (document_count - containing + 0.5) / (containing + 0.5))
            for doc, fields, length in zip(documents, token_fields, lengths):
                frequency = sum(tokens.count(term) for tokens in fields.values())
                if frequency:
                    normalization = self.k1 * (1 - self.b + self.b * length / average_length)
                    result[doc.id] += idf * frequency * (self.k1 + 1) / (frequency + normalization)
        return result
