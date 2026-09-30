from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .query import Clause, Query
from .segment import Segment, StoredDocument
from .synonyms import SynonymRules


@dataclass
class TermHit:
    term: str
    matched_as: str
    position: int

    @property
    def kind(self) -> str:
        return "original" if self.matched_as == self.term else "synonym"

    def to_dict(self) -> Dict[str, object]:
        return {
            "term": self.term,
            "matched_as": self.matched_as,
            "kind": self.kind,
            "position": self.position,
        }


@dataclass
class ClauseHit:
    clause: Clause
    field_name: str
    terms: List[TermHit]

    @property
    def mode(self) -> str:
        if not self.clause.expand:
            base = "strict"
        else:
            base = "expanded_phrase" if self.clause.phrase else "expanded_terms"
        if self.clause.phrase and self.clause.slop > 0:
            return f"{base}_slop"
        return base

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw": self.clause.raw,
            "field": self.clause.field,
            "phrase": self.clause.phrase,
            "expand": self.clause.expand,
            "mode": self.mode,
            "matched_field": self.field_name,
            "positions": [term.position for term in self.terms],
            "slop": self.clause.slop,
            "terms": [hit.to_dict() for hit in self.terms],
        }


@dataclass
class DocumentMatch:
    document: StoredDocument
    clause_hits: List[ClauseHit] = field(default_factory=list)

    def matched_tokens(self) -> List[str]:
        tokens: List[str] = []
        for hit in self.clause_hits:
            for term_hit in hit.terms:
                tokens.append(term_hit.matched_as)
        return tokens

    def explanation(
        self,
        score: float,
        term_contributions: Optional[Dict[str, object]] = None,
        rules_version: Optional[int] = None,
        excluded_rawhs: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        terms: Dict[str, Dict[str, object]] = {}
        for clause_hit in self.clause_hits:
            for term_hit in clause_hit.terms:
                entry = terms.setdefault(
                    term_hit.term,
                    {
                        "term": term_hit.term,
                        "matched_as": term_hit.matched_as,
                        "kind": term_hit.kind,
                        "occurrences": 0,
                        "positions": [],
                    },
                )
                entry["occurrences"] = int(entry["occurrences"]) + 1
                entry["positions"].append(term_hit.position)
                if entry["kind"] == "original":
                    entry["matched_as"] = term_hit.matched_as
                if term_contributions and term_hit.matched_as in term_contributions:
                    entry["score_contribution"] = term_contributions[term_hit.matched_as].get("contribution")
                    entry["matched_term_frequency"] = term_contributions[term_hit.matched_as].get("document_frequency")
        return {
            "score": score,
            "synonym_version": rules_version,
            "excluded_clauses": list(excluded_rawhs or []),
            "clauses": [hit.to_dict() for hit in self.clause_hits],
            "terms": list(terms.values()),
        }


@dataclass(frozen=True)
class _SegmentEntry:
    segment: Segment
    sequence: int


class SearchView:
    """An immutable queryable view: fixed visible documents, segments and rules.

    Both the live index and published snapshots are represented by a view,
    so old snapshots keep their own complete rule version and stay queryable
    regardless of later index changes.
    """

    def __init__(
        self,
        documents: Dict[str, StoredDocument],
        entries: List[_SegmentEntry],
        newest: Dict[str, int],
        rules: SynonymRules,
    ) -> None:
        self.documents = dict(documents)
        self.entries = list(entries)
        self.newest = dict(newest)
        self.rules = rules

    @classmethod
    def live(cls, segments: List[Segment], sequences: Dict[str, int], deletes: Dict[str, int], rules: SynonymRules) -> "SearchView":
        entries = [_SegmentEntry(segment, sequences[segment.segment_id]) for segment in segments]
        visible: Dict[str, StoredDocument] = {}
        newest: Dict[str, int] = {}
        for entry in sorted(entries, key=lambda item: item.sequence):
            for doc_id, doc in entry.segment.documents.items():
                if entry.sequence > deletes.get(doc_id, -1):
                    visible[doc_id] = doc
                    newest[doc_id] = entry.sequence
                elif newest.get(doc_id, -1) <= deletes.get(doc_id, -1):
                    visible.pop(doc_id, None)
        return cls(visible, entries, newest, rules)

    def run(self, query: Query) -> Dict[str, DocumentMatch]:
        matches: Dict[str, DocumentMatch] = {doc_id: DocumentMatch(doc) for doc_id, doc in self.documents.items()}
        for clause in query.clauses:
            clause_matches = self._match_clause(clause, query.rules or self.rules)
            if clause.exclude:
                for doc_id in clause_matches:
                    matches.pop(doc_id, None)
            else:
                for doc_id in list(matches):
                    hit = clause_matches.get(doc_id)
                    if hit is None:
                        del matches[doc_id]
                    else:
                        matches[doc_id].clause_hits.append(hit)
        return matches

    @staticmethod
    def excluded_rawhs(query: Query) -> List[str]:
        return [clause.raw for clause in query.clauses if clause.exclude]

    def _match_clause(self, clause: Clause, rules: SynonymRules) -> Dict[str, ClauseHit]:
        result: Dict[str, ClauseHit] = {}
        for entry in self.entries:
            field_names = [clause.field] if clause.field is not None else sorted(entry.segment.postings)
            for field_name in field_names:
                term_postings = entry.segment.postings.get(field_name, {})
                for doc_id in self._field_docs(term_postings, clause, rules):
                    if self.newest.get(doc_id) != entry.sequence or doc_id in result:
                        continue
                    hit = self._field_hit(field_name, term_postings, clause, doc_id, rules)
                    if hit is not None:
                        result[doc_id] = hit
        return result

    def _field_docs(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        clause: Clause,
        rules: SynonymRules,
    ) -> List[str]:
        first = clause.terms[0]
        candidates = set(term_postings.get(first, {}))
        if clause.expand:
            for synonym in rules.alternatives(first):
                candidates.update(term_postings.get(synonym, {}))
        return list(candidates)

    def _field_hit(
        self,
        field_name: str,
        term_postings: Dict[str, Dict[str, List[int]]],
        clause: Clause,
        doc_id: str,
        rules: SynonymRules,
    ) -> Optional[ClauseHit]:
        chains: List[List[TermHit]] = []
        if clause.phrase:
            chains = self._phrase_chains(term_postings, doc_id, clause, rules)
        else:
            chain = self._term_chain(term_postings, doc_id, clause, rules)
            if chain is not None:
                chains = [chain]
        if not chains:
            return None
        chosen = self._choose_chain(chains)
        return ClauseHit(clause, field_name, chosen)

    def _variants(self, term: str, clause: Clause, rules: SynonymRules) -> List[str]:
        if not clause.expand:
            return [term]
        return list(rules.candidates(term))

    def _term_chain(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        rules: SynonymRules,
    ) -> Optional[List[TermHit]]:
        """AND-of-terms within one field; repeated terms need distinct positions."""
        chain: List[TermHit] = []
        used: set[int] = set()
        for term in clause.terms:
            found = self._find_token(term_postings, doc_id, term, clause, rules, -1, used)
            if found is None:
                return None
            matched_as, position = found
            used.add(position)
            chain.append(TermHit(term, matched_as, position))
        return chain

    def _find_token(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        term: str,
        clause: Clause,
        rules: SynonymRules,
        after: int,
        used: set[int],
    ) -> Optional[Tuple[str, int]]:
        """Earliest token matching a term after ``after`` at an unused position.

        Original wording is preferred over a synonym when both are available
        at the same earliest position, so explanations are deterministic.
        """
        best: Optional[Tuple[int, int, str]] = None  # (position, original-first, token)
        for variant_index, variant in enumerate(self._variants(term, clause, rules)):
            for position in term_postings.get(variant, {}).get(doc_id, []):
                if position <= after or position in used:
                    continue
                rank = (position, 0 if variant == term else 1, variant_index)
                if best is None or rank < best:
                    best = rank + (variant,)
        if best is None:
            return None
        return best[2], best[0]

    def _phrase_chains(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        rules: SynonymRules,
    ) -> List[List[TermHit]]:
        """In-order phrase chains whose total gap is within the slop limit.

        Every matched token occupies its own position, so a repeated word
        can never reuse the same index, and each chain stays inside one
        field (the caller iterates fields independently).
        """
        chains: List[List[TermHit]] = []

        def extend(chain: List[TermHit], used: set[int], term_index: int, spent: int) -> None:
            if term_index == len(clause.terms):
                chains.append(list(chain))
                return
            term = clause.terms[term_index]
            last_pos = chain[-1].position if chain else -1
            options: List[Tuple[int, int, str]] = []
            for variant in self._variants(term, clause, rules):
                for position in term_postings.get(variant, {}).get(doc_id, []):
                    if position <= last_pos or position in used:
                        continue
                    gap = position - last_pos - 1 if chain else 0
                    if spent + gap > clause.slop:
                        continue
                    options.append((position, 0 if variant == term else 1, variant))
            options.sort(key=lambda item: (item[0], item[1]))
            seen_positions: set[int] = set()
            for position, _, variant in options:
                if position in seen_positions:
                    continue
                seen_positions.add(position)
                gap = position - last_pos - 1 if chain else 0
                used.add(position)
                chain.append(TermHit(term, variant, position))
                extend(chain, used, term_index + 1, spent + gap)
                chain.pop()
                used.discard(position)

        extend([], set(), 0, 0)
        return chains

    def _choose_chain(self, chains: List[List[TermHit]]) -> List[TermHit]:
        """Deterministic chain independent of posting/segment iteration order.

        Prefer the most original wording, then the tightest span, then the
        earliest start, so the same document always explains identically.
        """
        def score(chain: List[TermHit]) -> Tuple[Tuple[int, ...], int, Tuple[int, ...]]:
            kinds = tuple(0 if hit.kind == "original" else 1 for hit in chain)
            positions = tuple(hit.position for hit in chain)
            span = positions[-1] - positions[0] if positions else 0
            return kinds, span, positions

        return min(chains, key=score)
