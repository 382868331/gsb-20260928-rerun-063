import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from local_search.synonyms import SynonymRules

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def scan(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> List[str]:
    clauses = _parse(query_text)
    return [
        doc["id"]
        for doc in documents
        if all(_matches(doc["fields"], clause, rules) for clause in clauses)
    ]


def scan_explained(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    rules: Optional[SynonymRules] = None,
) -> Dict[str, List[Tuple[Optional[str], str, List[Tuple[str, str]]]]]:
    """Direct-scan reference for matching chains.

    Returns doc_id -> list of (matched_field, raw clause, term hits).
    """
    clauses = _parse(query_text)
    result: Dict[str, List[Tuple[Optional[str], str, List[Tuple[str, str]]]]] = {}
    for doc in documents:
        hits: List[Tuple[Optional[str], str, List[Tuple[str, str]]]] = []
        for field, terms, phrase, expand, raw in clauses:
            names = [field] if field is not None else list(doc["fields"])
            field_hit = None
            for name in names:
                tokens = TOKEN_RE.findall(str(doc["fields"].get(name, "")).lower())
                chains = _chains(tokens, terms, phrase, expand, rules)
                if chains:
                    chain = _choose_chain(chains)
                    field_hit = (name, raw, chain)
                    break
            if field_hit is None:
                break
            hits.append(field_hit)
        if len(hits) == len(clauses):
            result[doc["id"]] = hits
    return result


def _parse(query_text: str) -> List[Tuple[Optional[str], List[str], bool, bool, str]]:
    pattern = r'(~)?(?:(\w+):)?(?:"([^"]*)"|(\S+))'
    parsed = []
    for tilde, field, quoted, word in re.findall(pattern, query_text):
        if quoted != "":
            terms = TOKEN_RE.findall(quoted.lower())
            parsed.append((field or None, terms, True, bool(tilde), f'"{quoted}"'))
        else:
            terms = TOKEN_RE.findall(word.lower())
            parsed.append((field or None, terms, False, bool(tilde), word))
    return parsed


def _matches(
    fields: Dict[str, object],
    clause: Tuple[Optional[str], List[str], bool, bool, str],
    rules: Optional[SynonymRules],
) -> bool:
    field, terms, phrase, expand, _ = clause
    names = [field] if field is not None else list(fields)
    return any(
        _chains(TOKEN_RE.findall(str(fields.get(name, "")).lower()), terms, phrase, expand, rules)
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
) -> List[List[Tuple[str, str]]]:
    if phrase:
        chains: List[List[Tuple[str, str]]] = []
        for start in range(len(tokens) - len(terms) + 1):
            chain = []
            for offset, term in enumerate(terms):
                matched = None
                for variant in _variants(term, expand, rules):
                    if tokens[start + offset] == variant:
                        matched = variant
                        break
                if matched is None:
                    break
                chain.append((term, matched))
            if len(chain) == len(terms):
                chains.append(chain)
        return chains

    chain = []
    for term in terms:
        matched = None
        for variant in _variants(term, expand, rules):
            if variant in tokens:
                matched = variant
                break
        if matched is None:
            return []
        chain.append((term, matched))
    return [chain]


def _choose_chain(chains: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    def rank(chain: List[Tuple[str, str]]) -> Tuple[int, ...]:
        return tuple(0 if term == matched else 1 for term, matched in chain)

    return min(chains, key=rank)
