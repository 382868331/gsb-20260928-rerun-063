from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from .document import Document
from .query import And, Not, Or, PhraseQuery, Query, TermQuery
from .segment import Segment


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: Any
    document: Document


Ranker = Callable[[Hit, Segment], Any]


@dataclass(frozen=True)
class _Candidate:
    doc_id: str
    matched_terms: tuple[tuple[str, str], ...]


def bm25_score(candidate: _Candidate, segment: Segment) -> float:
    if segment.doc_count == 0 or not candidate.matched_terms:
        return 0.0
    total_field_length = sum(
        sum(lengths.values()) for lengths in segment.lengths.values()
    )
    average_length = max(1.0, total_field_length / segment.doc_count)
    counts = Counter(candidate.matched_terms)
    k1 = 1.2
    b = 0.75
    score = 0.0
    for (field, term), match_frequency in counts.items():
        document_frequency = len(segment.postings.get(field, {}).get(term, {}))
        inverse_frequency = (
            (segment.doc_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1.0
        )
        field_length = segment.lengths.get(candidate.doc_id, {}).get(field, 0)
        normalized_length = field_length / average_length
        score += inverse_frequency * (
            match_frequency * (k1 + 1)
            / (match_frequency + k1 * (1 - b + b * normalized_length))
        )
    return score


class Searcher:
    def __init__(self, segment: Segment):
        self.segment = segment

    def search(
        self,
        query: Query,
        ranker: Ranker | None = None,
        reverse: bool = True,
    ) -> list[Hit]:
        all_docs = set(self.segment.documents)
        matched, _ = self._evaluate(query, all_docs)
        hits: list[Hit] = []
        for candidate in matched:
            document = self.segment.documents[candidate.doc_id]
            default_hit = Hit(
                candidate.doc_id,
                bm25_score(candidate, self.segment),
                document,
            )
            if ranker is None:
                hits.append(default_hit)
            else:
                hits.append(
                    Hit(
                        candidate.doc_id,
                        ranker(default_hit, self.segment),
                        document,
                    )
                )
        return sorted(hits, key=lambda hit: (hit.score, hit.doc_id), reverse=reverse)

    def _evaluate(
        self,
        query: Query,
        all_docs: set[str],
    ) -> tuple[set[_Candidate], tuple[tuple[str, str], ...]]:
        if isinstance(query, TermQuery):
            wanted = query.clause_key()[2]
            if len(wanted) != 1:
                return set(), ()
            field, term = query.field, wanted[0]
            matches = {
                _Candidate(doc_id, ((field, term),))
                for doc_id in self.segment.postings.get(field, {}).get(term, {})
            }
            return matches, ((field, term),)
        if isinstance(query, PhraseQuery):
            wanted = query.clause_key()[2]
            if not wanted:
                return set(), ()
            matches = {
                _Candidate(
                    doc_id,
                    tuple((query.field, term) for term in wanted),
                )
                for doc_id in self._phrase_matches(query.field, list(wanted))
            }
            return matches, tuple((query.field, term) for term in wanted)
        if isinstance(query, And):
            candidates = {
                doc_id: _Candidate(doc_id, ())
                for doc_id in all_docs
            }
            terms: list[tuple[str, str]] = []
            for child in query.children:
                child_matches, child_terms = self._evaluate(child, all_docs)
                by_id = {candidate.doc_id: candidate for candidate in child_matches}
                for doc_id in list(candidates):
                    if doc_id not in by_id:
                        del candidates[doc_id]
                    else:
                        candidates[doc_id] = _Candidate(
                            doc_id,
                            candidates[doc_id].matched_terms + by_id[doc_id].matched_terms,
                        )
                terms.extend(child_terms)
            return set(candidates.values()), tuple(terms)
        if isinstance(query, Or):
            merged: dict[str, list[tuple[str, str]]] = {}
            terms = []
            for child in query.children:
                child_matches, child_terms = self._evaluate(child, all_docs)
                terms.extend(child_terms)
                for candidate in child_matches:
                    merged.setdefault(candidate.doc_id, []).extend(candidate.matched_terms)
            return {
                _Candidate(doc_id, tuple(matched_terms))
                for doc_id, matched_terms in merged.items()
            }, tuple(terms)
        if isinstance(query, Not):
            child_matches, _ = self._evaluate(query.child, all_docs)
            matched_ids = {candidate.doc_id for candidate in child_matches}
            return {
                _Candidate(doc_id, ()) for doc_id in all_docs - matched_ids
            }, ()
        raise TypeError(f"unsupported query type {type(query)!r}")

    def _phrase_matches(self, field: str, wanted_terms: list[str]) -> set[str]:
        field_postings = self.segment.postings.get(field, {})
        chains = [field_postings.get(term, {}) for term in wanted_terms]
        candidates = set(chains[0]) if chains else set()
        for chain in chains[1:]:
            candidates &= set(chain)
        matches: set[str] = set()
        for doc_id in candidates:
            for start in chains[0][doc_id]:
                if all(
                    start + offset in chains[offset][doc_id]
                    for offset in range(1, len(wanted_terms))
                ):
                    matches.add(doc_id)
                    break
        return matches
