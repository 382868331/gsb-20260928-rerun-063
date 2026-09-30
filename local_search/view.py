from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .query import Clause, Query
from .segment import Segment, StoredDocument
from .synonyms import SynonymRules


@dataclass
class TermHit:
    term: str
    matched_as: str

    @property
    def kind(self) -> str:
        return "original" if self.matched_as == self.term else "synonym"

    def to_dict(self) -> Dict[str, object]:
        return {"term": self.term, "matched_as": self.matched_as, "kind": self.kind}


@dataclass
class ClauseHit:
    clause: Clause
    field_name: str
    terms: List[TermHit]

    @property
    def mode(self) -> str:
        if not self.clause.expand:
            return "strict"
        return "expanded_phrase" if self.clause.phrase else "expanded_terms"

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw": self.clause.raw,
            "field": self.clause.field,
            "phrase": self.clause.phrase,
            "expand": self.clause.expand,
            "mode": self.mode,
            "matched_field": self.field_name,
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

    def explanation(self, score: float, term_contributions: Optional[Dict[str, object]] = None, rules_version: Optional[int] = None) -> Dict[str, object]:
        terms: Dict[str, Dict[str, object]] = {}
        for clause_hit in self.clause_hits:
            for term_hit in clause_hit.terms:
                entry = terms.setdefault(
                    term_hit.term,
                    {"term": term_hit.term, "matched_as": term_hit.matched_as, "kind": term_hit.kind, "occurrences": 0},
                )
                entry["occurrences"] = int(entry["occurrences"]) + 1
                if entry["kind"] == "original":
                    entry["matched_as"] = term_hit.matched_as
                if term_contributions and term_hit.matched_as in term_contributions:
                    entry["score_contribution"] = term_contributions[term_hit.matched_as].get("contribution")
                    entry["matched_term_frequency"] = term_contributions[term_hit.matched_as].get("document_frequency")
        return {
            "score": score,
            "synonym_version": rules_version,
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
            for doc_id in list(matches):
                hit = clause_matches.get(doc_id)
                if hit is None:
                    del matches[doc_id]
                else:
                    matches[doc_id].clause_hits.append(hit)
        return matches

    def _match_clause(self, clause: Clause, rules: SynonymRules) -> Dict[str, ClauseHit]:
        result: Dict[str, ClauseHit] = {}
        for entry in self.entries:
            field_names = [clause.field] if clause.field is not None else list(entry.segment.postings)
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

    def _term_chain(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        rules: SynonymRules,
    ) -> Optional[List[TermHit]]:
        chain: List[TermHit] = []
        for term in clause.terms:
            matched_as = self._present_token(term_postings, doc_id, term, clause.expand, rules)
            if matched_as is None:
                return None
            chain.append(TermHit(term, matched_as))
        return chain

    def _present_token(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        term: str,
        expand: bool,
        rules: SynonymRules,
    ) -> Optional[str]:
        if doc_id in term_postings.get(term, {}):
            return term
        if expand:
            for synonym in rules.alternatives(term):
                if doc_id in term_postings.get(synonym, {}):
                    return synonym
        return None

    def _phrase_chains(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        rules: SynonymRules,
    ) -> List[List[TermHit]]:
        start_positions = set(term_postings.get(clause.terms[0], {}).get(doc_id, []))
        if clause.expand:
            for synonym in rules.alternatives(clause.terms[0]):
                start_positions.update(term_postings.get(synonym, {}).get(doc_id, []))
        chains: List[List[TermHit]] = []
        for start in start_positions:
            chain = self._chain_at(term_postings, doc_id, clause, start, rules)
            if chain is not None:
                chains.append(chain)
        return chains

    def _chain_at(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        start: int,
        rules: SynonymRules,
    ) -> Optional[List[TermHit]]:
        chain: List[TermHit] = []
        used_positions: set[int] = set()
        for offset, term in enumerate(clause.terms):
            position = start + offset
            if position in used_positions:
                return None
            token = self._token_at(term_postings, doc_id, term, position, clause.expand, rules)
            if token is None:
                return None
            used_positions.add(position)
            chain.append(TermHit(term, token))
        return chain

    def _token_at(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        term: str,
        position: int,
        expand: bool,
        rules: SynonymRules,
    ) -> Optional[str]:
        if position in term_postings.get(term, {}).get(doc_id, []):
            return term
        if expand:
            for synonym in rules.alternatives(term):
                if position in term_postings.get(synonym, {}).get(doc_id, []):
                    return synonym
        return None

    def _choose_chain(self, chains: List[List[TermHit]]) -> List[TermHit]:
        def score(chain: List[TermHit]) -> Tuple[int, ...]:
            return tuple(0 if hit.kind == "original" else 1 for hit in chain)

        return min(chains, key=score)
