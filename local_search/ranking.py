import math
from typing import Dict, List, Protocol, Sequence

from .analysis import field_tokens
from .query import Query
from .segment import StoredDocument
from .view import DocumentMatch


class Ranker(Protocol):
    def score(
        self,
        matches: Dict[str, DocumentMatch],
        query: Query,
    ) -> Dict[str, Dict[str, object]]:
        ...


class ConstantScoreRanker:
    def score(
        self,
        matches: Dict[str, DocumentMatch],
        query: Query,
    ) -> Dict[str, Dict[str, object]]:
        return {doc_id: {"score": 0.0, "term_contributions": {}} for doc_id in matches}


class BM25Ranker:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        matches: Dict[str, DocumentMatch],
        query: Query,
    ) -> Dict[str, Dict[str, object]]:
        documents = [match.document for match in matches.values()]
        document_count = len(matches) or 1
        terms = list(dict.fromkeys(token for match in matches.values() for token in match.matched_tokens()))
        token_fields = [field_tokens(doc.fields) for doc in documents]
        lengths = [sum(len(tokens) for tokens in fields.values()) for fields in token_fields]
        average_length = sum(lengths) / document_count or 1.0
        result: Dict[str, Dict[str, object]] = {
            doc.id: {"score": 0.0, "term_contributions": {}} for doc in documents
        }

        for term in terms:
            containing = sum(
                1 for fields in token_fields if any(term in tokens for tokens in fields.values())
            )
            idf = math.log(1 + (document_count - containing + 0.5) / (containing + 0.5))
            for doc, fields, length in zip(documents, token_fields, lengths):
                frequency = sum(tokens.count(term) for tokens in fields.values())
                if frequency:
                    normalization = self.k1 * (1 - self.b + self.b * length / average_length)
                    contribution = idf * frequency * (self.k1 + 1) / (frequency + normalization)
                    result[doc.id]["score"] = float(result[doc.id]["score"]) + contribution
                    result[doc.id]["term_contributions"][term] = {
                        "idf": idf,
                        "document_frequency": frequency,
                        "contribution": contribution,
                    }
        return result


def ranker_descriptor(ranker: Ranker) -> Dict[str, object]:
    if isinstance(ranker, ConstantScoreRanker):
        return {"type": "ConstantScoreRanker", "module": "local_search.ranking", "params": {}}
    if isinstance(ranker, BM25Ranker):
        return {
            "type": "BM25Ranker",
            "module": "local_search.ranking",
            "params": {"k1": ranker.k1, "b": ranker.b},
        }
    return {
        "type": ranker.__class__.__name__,
        "module": ranker.__class__.__module__,
        "params": {},
        "persistent": False,
    }


def ranker_from_descriptor(descriptor: Dict[str, object]) -> Ranker:
    name = str(descriptor.get("type"))
    params = descriptor.get("params", {}) or {}
    if name == "ConstantScoreRanker":
        return ConstantScoreRanker()
    if name == "BM25Ranker":
        return BM25Ranker(k1=float(params.get("k1", 1.2)), b=float(params.get("b", 0.75)))  # type: ignore[union-attr]
    raise ValueError(f"Unsupported persistent ranker: {name}")


def ranker_matches_descriptor(ranker: Ranker, descriptor: Dict[str, object]) -> bool:
    current = ranker_descriptor(ranker)
    return current.get("type") == descriptor.get("type") and current.get("params") == descriptor.get("params")
