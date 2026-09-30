import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from local_search.synonyms import SynonymRules

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def scan(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> List[str]:
    from local_search.query import parse_query

    clauses = parse_query(query_text).clauses
    return [
        doc["id"]
        for doc in documents
        if all(
            (not _matches(doc["fields"], clause, rules)) if clause.exclude
            else _matches(doc["fields"], clause, rules)
            for clause in clauses
        )
    ]


def scan_explained(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> Dict[str, List[Tuple[Optional[str], str, List[Tuple[str, str]]]]]:
    """Direct-scan reference for matching chains.

    Returns doc_id -> list of (matched_field, raw clause, term hits).
    """
    from local_search.query import parse_query

    clauses = parse_query(query_text).clauses
    result: Dict[str, List[Tuple[Optional[str], str, List[Tuple[str, str]]]]] = {}
    for doc in documents:
        hits: List[Tuple[Optional[str], str, List[Tuple[str, str]]]] = []
        for clause in clauses:
            if clause.exclude:
                if _matches(doc["fields"], clause, rules):
                    break
                continue
            field, terms, phrase, expand, raw, slop = (
                clause.field, clause.terms, clause.phrase, clause.expand, clause.raw, clause.slop,
            )
            names = [field] if field is not None else sorted(doc["fields"])
            field_hit = None
            for name in names:
                tokens = TOKEN_RE.findall(str(doc["fields"].get(name, "")).lower())
                chains = _chains(tokens, terms, phrase, expand, rules, slop)
                if chains:
                    chain = _choose_chain(chains)
                    field_hit = (name, raw, chain)
                    break
            if field_hit is None:
                break
            hits.append(field_hit)
        if len(hits) == len([clause for clause in clauses if not clause.exclude]):
            result[doc["id"]] = hits
    return result


def _matches(
    fields: Dict[str, object],
    clause,
    rules: Optional[SynonymRules],
) -> bool:
    names = [clause.field] if clause.field is not None else sorted(fields)
    return any(
        _chains(
            TOKEN_RE.findall(str(fields.get(name, "")).lower()),
            clause.terms,
            clause.phrase,
            clause.expand,
            rules,
            clause.slop,
        )
        for name in names
    )


def _variants(term: str, expand: bool, rules: Optional[SynonymRules]) -> List[str]:
    if not expand or rules is None:
        return [term]
    return list(rules.candidates(term))


def _chains(
    tokens: Sequence[str],
    terms: Sequence[str],
    phrase: bool,
    expand: bool,
    rules: Optional[SynonymRules],
    slop: int = 0,
) -> List[List[Tuple[str, str]]]:
    if phrase:
        chains: List[List[Tuple[str, str, int]]] = []

        def extend(chain, used, term_index, spent):
            if term_index == len(terms):
                chains.append(list(chain))
                return
            term = terms[term_index]
            last_pos = chain[-1][2] if chain else -1
            options = []
            for variant in _variants(term, expand, rules):
                for position, token in enumerate(tokens):
                    if position <= last_pos or position in used:
                        continue
                    if token != variant:
                        continue
                    gap = position - last_pos - 1 if chain else 0
                    if spent + gap <= slop:
                        options.append((position, 0 if variant == term else 1, variant))
            seen = set()
            for position, _, variant in sorted(options):
                if position in seen:
                    continue
                seen.add(position)
                gap = position - last_pos - 1 if chain else 0
                used.add(position)
                chain.append((term, variant, position))
                extend(chain, used, term_index + 1, spent + gap)
                chain.pop()
                used.discard(position)

        extend([], set(), 0, 0)
        return chains

    chain = []
    used = set()
    for term in terms:
        best = None
        for variant in _variants(term, expand, rules):
            for position, token in enumerate(tokens):
                if token == variant and position not in used:
                    rank = (position, 0 if variant == term else 1)
                    if best is None or rank < (best[0], best[1]):
                        best = (position, 0 if variant == term else 1, variant)
        if best is None:
            return []
        used.add(best[0])
        chain.append((term, best[2], best[0]))
    return [chain]


def _choose_chain(chains: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    def rank(chain: List[Tuple[str, str]]) -> Tuple[Tuple[int, ...], int, Tuple[int, ...]]:
        kinds = tuple(0 if term == matched else 1 for term, matched, _ in chain)
        positions = tuple(position for _, _, position in chain)
        span = positions[-1] - positions[0] if positions else 0
        return kinds, span, positions

    return min(chains, key=rank)
