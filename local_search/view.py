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
    synonym_group: Optional[Tuple[str, ...]] = None

    @property
    def kind(self) -> str:
        return "original" if self.matched_as == self.term else "synonym"

    def to_dict(self) -> Dict[str, object]:
        return {
            "term": self.term,
            "matched_as": self.matched_as,
            "kind": self.kind,
            "position": self.position,
            "synonym_group": list(self.synonym_group) if self.synonym_group else None,
        }


@dataclass
class ClauseHit:
    clause: Clause
    field_name: str
    terms: List[TermHit]

    @property
    def mode(self) -> str:
        if self.clause.negated:
            return "excluded"
        if not self.clause.expand:
            return "strict"
        return "expanded_phrase" if self.clause.phrase else "expanded_terms"

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw": self.clause.raw,
            "field": self.clause.field,
            "phrase": self.clause.phrase,
            "expand": self.clause.expand,
            "negated": self.clause.negated,
            "slop": self.clause.slop,
            "mode": self.mode,
            "matched_field": self.field_name,
            "terms": [hit.to_dict() for hit in self.terms],
        }


@dataclass
class DocumentMatch:
    document: StoredDocument
    clause_hits: List[ClauseHit] = field(default_factory=list)
    excluded_clauses: List[str] = field(default_factory=list)

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
                        "synonym_group": term_hit.synonym_group,
                    },
                )
                entry["occurrences"] = int(entry["occurrences"]) + 1
                entry["positions"].append(term_hit.position)
                if term_hit.synonym_group is not None:
                    entry["synonym_group"] = list(term_hit.synonym_group)
                if term_contributions and term_hit.matched_as in term_contributions:
                    entry["score_contribution"] = term_contributions[term_hit.matched_as].get("contribution")
                    entry["matched_term_frequency"] = term_contributions[term_hit.matched_as].get("document_frequency")
        return {
            "score": score,
            "synonym_version": rules_version,
            "clauses": [hit.to_dict() for hit in self.clause_hits],
            "exclusions": list(self.excluded_clauses),
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
    def live(
        cls,
        segments: List[Segment],
        sequences: Dict[str, int],
        deletes: Dict[str, int],
        rules: SynonymRules,
    ) -> "SearchView":
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
        rules = query.rules or self.rules
        matches: Dict[str, DocumentMatch] = {doc_id: DocumentMatch(doc) for doc_id, doc in self.documents.items()}
        for clause in query.clauses:
            clause_matches = self._match_clause(clause, rules)
            if clause.negated:
                for doc_id in clause_matches:
                    matches.pop(doc_id, None)
                for match in matches.values():
                    match.excluded_clauses.append(clause.raw)
                continue
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
            field_names = [clause.field] if clause.field is not None else sorted(entry.segment.postings)
            for doc_id, doc in entry.segment.documents.items():
                if self.newest.get(doc_id) != entry.sequence or doc_id in result:
                    continue
                best: Optional[ClauseHit] = None
                for field_name in field_names:
                    term_postings = entry.segment.postings.get(field_name, {})
                    if doc_id not in self._field_docs(term_postings, clause, rules):
                        continue
                    hit = self._field_hit(field_name, term_postings, clause, doc_id, rules)
                    if hit is not None and (best is None or self._chain_rank(hit) < self._chain_rank(best)):
                        best = hit
                if best is not None:
                    result[doc_id] = best
        return result

    def _chain_rank(self, hit: ClauseHit) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        kinds = tuple(0 if term.kind == "original" else 1 for term in hit.terms)
        positions = tuple(term.position for term in hit.terms)
        return (kinds, positions, positions[-1] - positions[0])

    def _field_docs(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        clause: Clause,
        rules: SynonymRules,
    ) -> List[str]:
        candidate_sets: List[set[str]] = []
        for term in clause.terms:
            candidates = set(term_postings.get(term, {}))
            if clause.expand:
                for synonym in rules.alternatives(term):
                    candidates.update(term_postings.get(synonym, {}))
            candidate_sets.append(candidates)
        return sorted(set.intersection(*candidate_sets)) if candidate_sets else []

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
        return ClauseHit(clause, field_name, self._choose_chain(chains))

    def _variants(self, term: str, expand: bool, rules: SynonymRules) -> List[Tuple[str, Optional[Tuple[str, ...]]]]:
        variants: List[Tuple[str, Optional[Tuple[str, ...]]]] = [(term, None)]
        if expand:
            for synonym in rules.alternatives(term):
                variants.append((synonym, rules.group_for(term, synonym)))
        return variants

    def _term_chain(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        rules: SynonymRules,
    ) -> Optional[List[TermHit]]:
        # Every matched token must consume a distinct position: repeated
        # query terms (and synonym overlaps) cannot share one occurrence.
        # Search all assignments so a locally attractive choice cannot
        # starve a later term; the canonical chain is picked afterwards.
        variants = [self._variants(term, clause.expand, rules) for term in clause.terms]
        chains: List[List[TermHit]] = []

        def extend(index: int, chain: List[TermHit], used: set[int]) -> None:
            if index == len(clause.terms):
                chains.append(list(chain))
                return
            term = clause.terms[index]
            seen: set[Tuple[str, int]] = set()
            for token, group in variants[index]:
                for position in term_postings.get(token, {}).get(doc_id, []):
                    if position in used or (token, position) in seen:
                        continue
                    seen.add((token, position))
                    chain.append(TermHit(term, token, position, group))
                    used.add(position)
                    extend(index + 1, chain, used)
                    used.remove(position)
                    chain.pop()

        extend(0, [], set())
        return self._choose_chain(chains) if chains else None

    def _phrase_chains(
        self,
        term_postings: Dict[str, Dict[str, List[int]]],
        doc_id: str,
        clause: Clause,
        rules: SynonymRules,
    ) -> List[List[TermHit]]:
        variants = [self._variants(term, clause.expand, rules) for term in clause.terms]
        position_lists = [
            sorted(
                {
                    (position, token, group)
                    for token, group in variant_options
                    for position in term_postings.get(token, {}).get(doc_id, [])
                }
            )
            for variant_options in variants
        ]
        slop = clause.slop or 0
        chains: List[List[TermHit]] = []

        def extend(index: int, previous: int, budget: int, chain: List[TermHit], used: set[int]) -> None:
            if index == len(clause.terms):
                chains.append(list(chain))
                return
            term = clause.terms[index]
            for position, token, group in position_lists[index]:
                if position <= previous or position in used:
                    continue
                gap = position - previous - 1
                if gap > budget:
                    break
                chain.append(TermHit(term, token, position, group))
                used.add(position)
                extend(index + 1, position, budget - gap, chain, used)
                used.remove(position)
                chain.pop()

        first = clause.terms[0]
        for position, token, group in position_lists[0]:
            extend(1, position, slop, [TermHit(first, token, position, group)], {position})
        return chains

    def _choose_chain(self, chains: List[List[TermHit]]) -> List[TermHit]:
        def score(chain: List[TermHit]) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, ...]:
            kinds = tuple(0 if hit.kind == "original" else 1 for hit in chain)
            positions = tuple(hit.position for hit in chain)
            return (kinds, positions, positions[-1] - positions[0])

        return min(chains, key=score)
