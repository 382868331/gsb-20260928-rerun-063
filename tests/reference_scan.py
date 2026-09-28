import re
from typing import Dict, Iterable, List, Optional, Sequence


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def scan(documents: Iterable[Dict[str, object]], query_text: str) -> List[str]:
    return [
        doc["id"]
        for doc in documents
        if all(_matches(doc["fields"], clause) for clause in _parse(query_text))
    ]


def _parse(query_text: str) -> List[tuple[Optional[str], List[str], bool]]:
    pattern = r'(?:(\w+):)?(?:"([^"]*)"|(\S+))'
    return [
        (field or None, TOKEN_RE.findall((phrase or word).lower()), bool(phrase))
        for field, phrase, word in re.findall(pattern, query_text)
    ]


def _matches(fields: Dict[str, object], clause: tuple[Optional[str], List[str], bool]) -> bool:
    field, terms, phrase = clause
    names = [field] if field is not None else list(fields)
    return any(_field_matches(fields.get(name, ""), terms, phrase) for name in names)


def _field_matches(value: object, terms: Sequence[str], phrase: bool) -> bool:
    tokens = TOKEN_RE.findall(str(value).lower())
    if not phrase:
        return all(term in tokens for term in terms)
    return any(tokens[start : start + len(terms)] == terms for start in range(len(tokens) - len(terms) + 1))
