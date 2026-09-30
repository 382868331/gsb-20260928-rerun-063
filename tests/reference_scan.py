import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from local_search.query import Clause, parse_query
from local_search.synonyms import SynonymRules

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class RefTermHit:
    term: str
    matched_as: str
    position: int
    group: Optional[Tuple[str, ...]]

    @property
    def kind(self) -> str:
        return "original" if self.matched_as == self.term else "synonym"


def scan(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> List[str]:
    return [doc_id for doc_id, _ in scan_full(documents, query_text, rules)]


def scan_full(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> List[Tuple[str, List[Tuple[str, Clause, List[RefTermHit]]]]]:
    """Direct positional scan, returning surviving docs with positive hits."""
    query = parse_query(query_text)
    result = []
    for document in documents:
        doc = document
        positive: List[Tuple[str, Clause, List[RefTermHit]]] = []
        alive = True
        for clause in query.clauses:
            names = [clause.field] if clause.field is not None else sorted(doc["fields"])
            field_hit = None
            for name in names:
                tokens = TOKEN_RE.findall(str(doc["fields"].get(name, "")).lower())
                chains = _chains(tokens, clause, rules)
                if chains:
                    field_hit = (name, clause, _choose(chains))
                    break
            if clause.negated:
                if field_hit is not None:
                    alive = False
                    break
            elif field_hit is None:
                alive = False
                break
            else:
                positive.append(field_hit)
        if alive:
            result.append((doc["id"], positive))
    return result


def scan_explained(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> Dict[str, List[Tuple[str, str, List[Tuple[str, str, int]]]]]:
    """Direct-scan reference for matching chains with positions."""
    return {
        doc_id: [
            (
                field_name,
                clause.raw,
                [(hit.term, hit.matched_as, hit.position) for hit in chain],
            )
            for field_name, clause, chain in hits
        ]
        for doc_id, hits in scan_full(documents, query_text, rules)
    }


def _variants(
    term: str, clause: Clause, rules: Optional[SynonymRules]
) -> List[Tuple[str, Optional[Tuple[str, ...]]]]:
    options = [(term, None)]
    if clause.expand and rules is not None:
        for synonym in rules.alternatives(term):
            options.append((synonym, rules.group_for(term, synonym)))
    return options


def _chains(
    tokens: Sequence[str],
    clause: Clause,
    rules: Optional[SynonymRules],
) -> List[List[RefTermHit]]:
    options = [_variants(term, clause, rules) for term in clause.terms]
    position_lists = [
        sorted(
            {
                (index, token, group)
                for token, group in variant_options
                for index in [i for i, value in enumerate(tokens) if value == token]
            }
        )
        for variant_options in options
    ]
    chains: List[List[RefTermHit]] = []

    if clause.phrase:
        slop = clause.slop or 0

        def extend_phrase(
            index: int, previous: int, budget: int, chain: List[RefTermHit], used: set[int]
        ) -> None:
            if index == len(clause.terms):
                chains.append(list(chain))
                return
            for position, token, group in position_lists[index]:
                if position <= previous or position in used:
                    continue
                gap = position - previous - 1
                if gap > budget:
                    break
                chain.append(RefTermHit(clause.terms[index], token, position, group))
                used.add(position)
                extend_phrase(index + 1, position, budget - gap, chain, used)
                used.remove(position)
                chain.pop()

        for position, token, group in position_lists[0]:
            extend_phrase(
                1,
                position,
                slop,
                [RefTermHit(clause.terms[0], token, position, group)],
                {position},
            )
        return chains

    def extend_terms(index: int, chain: List[RefTermHit], used: set[int]) -> None:
        if index == len(clause.terms):
            chains.append(list(chain))
            return
        seen: set[Tuple[str, int]] = set()
        for position, token, group in position_lists[index]:
            if position in used or (token, position) in seen:
                continue
            seen.add((token, position))
            chain.append(RefTermHit(clause.terms[index], token, position, group))
            used.add(position)
            extend_terms(index + 1, chain, used)
            used.remove(position)
            chain.pop()

    extend_terms(0, [], set())
    return chains


def _choose(chains: List[List[RefTermHit]]) -> List[RefTermHit]:
    def score(chain: List[RefTermHit]) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        kinds = tuple(0 if hit.kind == "original" else 1 for hit in chain)
        positions = tuple(hit.position for hit in chain)
        return (kinds, positions, positions[-1] - positions[0])

    return min(chains, key=score)
