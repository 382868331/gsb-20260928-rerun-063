from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from local_search.analysis import SynonymRules, field_tokens
from local_search.matcher import ClauseHit, match_query
from local_search.query import parse_query


def scan(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    synonyms: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[str]:
    """Direct in-memory scan used as the reference implementation."""
    rules = SynonymRules.from_mapping(synonyms or {}, version=1)
    query = parse_query(query_text)
    hits = []
    for doc in documents:
        tokenized = field_tokens(doc["fields"])
        if match_query(doc["fields"], tokenized, query, rules) is not None:
            hits.append(doc["id"])
    return hits


def scan_explained(
    documents: Iterable[Dict[str, object]],
    query_text: str,
    synonyms: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Tuple[str, List[ClauseHit]]]:
    rules = SynonymRules.from_mapping(synonyms or {}, version=1)
    query = parse_query(query_text)
    output: List[Tuple[str, List[ClauseHit]]] = []
    for doc in documents:
        tokenized = field_tokens(doc["fields"])
        matched = match_query(doc["fields"], tokenized, query, rules)
        if matched is not None:
            output.append((doc["id"], matched))
    return output
